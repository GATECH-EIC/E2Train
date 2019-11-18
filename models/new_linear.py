"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Function

from models.quantize import calculate_qparams, quantize, quantize_grad
from models.predictive import mixing_output, quant_weight, quant_bias

# Inherit from Function
class PredictiveLinearFunction(Function):

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

        # Calculate gradients w.r.t. weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.bmm(grad_output.unsqueeze(2), input.unsqueeze(1))
            grad_weight = grad_weight.sum(dim=0).sign()

        # Calculate gradients w.r.t. bias, if needed
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = bias.new_ones(bias.shape) * torch.sum(grad_output, dim=0)
            grad_bias.sign_()

        return grad_input, grad_weight, grad_bias


class PredictiveLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, # the above two lines are the same as Conv2d
                 num_bits_weight=8, num_bits_bias=16, # weight and bias precision
                 input_signed=False, # whether the input is signed or unsigned
                 predictive_forward=True, predictive_backward=True,
                 msb_bits=4, msb_bits_weight=4, msb_bits_bias=8, msb_bits_grad=16,
                 threshold=5e-5, sparsify=False, sign=False, writer=None):

        super(PredictiveLinear, self).__init__(in_features, out_features, bias)

        self.num_bits_weight = num_bits_weight
        self.num_bits_bias = num_bits_bias
        self.input_signed = input_signed
        self.predictive_forward = predictive_forward
        self.predictive_backward = predictive_backward
        self.msb_bits = msb_bits
        self.msb_bits_weight = msb_bits_weight
        self.msb_bits_bias = msb_bits_bias
        self.msb_bits_grad = msb_bits_grad
        self.threshold = threshold
        self.sparsify = sparsify
        self.sign = sign

        if not self.predictive_backward:
            self.msb_bits_grad = None
            if not self.predictive_forward:
                self.msb_bits = self.msb_bits_weight = None
            else:
                assert self.msb_bits is not None and self.msb_bits_weight is not None

    def forward(self, input):
        # See the autograd section for explanation of what happens here.

        # Quantize `input` to get `msb_input`
        if self.msb_bits is not None:
            msb_input_qparams = calculate_qparams(input.detach(), num_bits=self.msb_bits,
                                                  flatten_dims=(1,-1), reduce_dim=0)
            msb_input = quantize(input, qparams=msb_input_qparams, signed=self.input_signed)
        else:
            msb_input = None

        # Quantize weight
        weights = quant_weight(
            self.weight, num_bits_weight=self.num_bits_weight,
            msb_bits_weight=self.msb_bits_weight, threshold=self.threshold,
            sparsify=self.sparsify, sign=self.sign)
        q_weight = weights[0]
        msb_weight = weights[1] if len(weights) > 1 else None

        # Quantize bias
        if self.bias is not None:
            biases = quant_bias(
                self.bias, num_bits_bias=self.num_bits_bias,
                msb_bits_bias=self.msb_bits_bias, threshold=self.threshold,
                sparsify=self.sparsify, sign=self.sign)
            q_bias = biases[0]
            msb_bias = biases[1] if len(biases) > 1 else None
        else:
            q_bias = None

        # Q-branch
        q_output = F.linear(input, q_weight, q_bias)
        # MSB-branch
        if self.predictive_forward and msb_input is not None and msb_weight is not None and msb_bias is not None:
            msb_output = F.linear(msb_input, msb_weight, msb_bias)
        else:
            msb_output = None

        # Mixing `q_output` and `msb_output`
        output = mixing_output(q_output, msb_output, self.msb_bits_grad)

        return output

        # Quantize weight
        weight_qparams = calculate_qparams(self.weight, num_bits=self.num_bits_weight,
                                           flatten_dims=(1,-1), reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams, signed=True)



        # Calculate convolution with (quantized) input, quantized weight and bias.
        # When performing backward(), we will use the predictive SignSGD algorithm.
        # The algorithm will use the msb precision and a threshold to deside which
        # gradient will be calculated accurately.
        return PredictiveSignLinearFunction.apply(input, qweight, qbias)
