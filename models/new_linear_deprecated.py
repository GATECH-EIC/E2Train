"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Function

from models.quantize import calculate_qparams, quantize, quantize_grad

# Inherit from Function
class PredictiveSignLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    # NOTE that here input, weight and bias can be quantized beforehand.
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        with torch.no_grad():
            output = F.linear(input, weight, bias)
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
            grad_input = F.linear(grad_output, torch.transpose(weight,0,1))
            # grad_input = torch.nn.grad.conv2d_input(
            #     input.shape, weight, grad_output,
            #     ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        # Calculate gradients w.r.t. weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.bmm(grad_output.unsqueeze(2), input.unsqueeze(1))
            grad_weight = grad_weight.sum(dim=0).sign()

        # Calculate gradients w.r.t. bias, if needed
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = bias.new_ones(bias.shape) * torch.sum(grad_output, dim=0)
            grad_bias.sign_()

        return grad_input, grad_weight, grad_bias


class PredictiveSignLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, # the above two lines are the same as Conv2d
                 num_bits_weight=8, num_bits_bias=16): # how the weight and the bias will be quantized

        super(PredictiveSignLinear, self).__init__(in_features, out_features, bias)

        self.num_bits_weight = num_bits_weight
        self.num_bits_bias = num_bits_bias

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
        return PredictiveSignLinearFunction.apply(input, qweight, qbias)
