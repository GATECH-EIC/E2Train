"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function

from models.quantize import calculate_qparams, quantize, quantize_grad


# Inherit from Function
class PredictiveSignConv2dFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    # NOTE that here input, weight and bias can be quantized beforehand.
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups,
                input_signed, msb_bits, msb_bits_weight, msb_bits_grad, threshold):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_signed = input_signed
        ctx.msb_bits = msb_bits
        ctx.msb_bits_weight = msb_bits_weight
        ctx.msb_bits_grad = msb_bits_grad
        ctx.threshold = threshold

        with torch.no_grad():
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # Calculate gradients w.r.t. input
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        # Calculate gradients w.r.t. weight
        if ctx.needs_input_grad[1]:
            # First quantize input, weight and grads to msb precision
            # Input
            msb_input_qparams = calculate_qparams(
                input, num_bits=ctx.msb_bits,
                flatten_dims=(1,-1), reduce_dim=0)
            msb_input = quantize(input, qparams=msb_input_qparams, signed=ctx.input_signed)
            # Weight
            msb_weight_qparams = calculate_qparams(
                weight, num_bits=ctx.msb_bits_weight,
                flatten_dims=(1,-1), reduce_dim=None)
            msb_weight = quantize(weight, qparams=msb_weight_qparams, signed=True)
            # Grads
            msb_grad_output_qparams = calculate_qparams(
                grad_output, num_bits=ctx.msb_bits_grad,
                flatten_dims=(1,-1), reduce_dim=0)
            msb_grad_output = quantize(grad_output, qparams=msb_grad_output_qparams, signed=True)

            # NOTE: Two branches here:
            #   - The first branch uses original (input, grad_output)
            #   - The second branch uses (msb_input, msb_grad)
            acc_grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
            pred_grad_weight = torch.nn.grad.conv2d_weight(
                msb_input, weight.shape, msb_grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

            # Combine the accurate and predictive weight grad based on `ctx.threshold`
            small_mag_locs = (pred_grad_weight.abs() <= ctx.threshold).float()
            grad_weight = small_mag_locs * acc_grad_weight + (1-small_mag_locs) * pred_grad_weight

            # Take signs of the gradient
            grad_weight.sign_()

        # Calculate gradients w.r.t. bias, if needed
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = bias.new_ones(bias.shape) * torch.sum(grad_output, dim=(0, 2, 3))
            grad_bias.sign_()

        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None, None, None, None, None)


class PredictiveSignConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, # the above two lines are the same as Conv2d
                 num_bits_weight=8, num_bits_bias=16, # how the weight and the bias will be quantized
                 input_signed=False, # whether the input is signed or unsigned
                 msb_bits=4, msb_bits_weight=4, msb_bits_grad=16, threshold=5e-5): # used in `backward()`
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PredictiveSignConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

        self.num_bits_weight = num_bits_weight
        self.num_bits_bias = num_bits_bias
        self.input_signed = input_signed
        self.msb_bits = msb_bits
        self.msb_bits_weight = msb_bits_weight
        self.msb_bits_grad = msb_bits_grad
        self.threshold = threshold

    def forward(self, input):
        # See the autograd section for explanation of what happens here.

        # Quantize weight
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits_weight,
                                           flatten_dims=(1,-1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams, signed=True)

        # Quantize bias
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_bias,
                             flatten_dims=(0,-1), signed=True)
        else:
            qbias = None

        # Calculate convolution with (quantized) input, quantized weight and bias.
        # When performing backward(), we will use the predictive SignSGD algorithm.
        # The algorithm will use the msb precision and a threshold to deside which
        # gradient will be calculated accurately.
        return PredictiveSignConv2dFunction.apply(
            input, qweight, qbias, self.stride, self.padding, self.dilation, self.groups,
            self.input_signed, self.msb_bits, self.msb_bits_weight, self.msb_bits_grad, self.threshold)
