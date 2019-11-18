"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function

from models.quantize import quantize, quantize_grad, Quantize
from models.quantize import efficient_quantize, EfficientQuantize
from models.predictive import mixing_output, quant_weight, efficient_quant_weight


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


# Inherit from Function
class PredictiveConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, # NOTE: CONV layer has no bias by default # the above two lines are the same as Conv2d
                 num_bits=8, num_bits_weight=8, num_bits_grad=8,
                 biprecision=True, input_signed=False,
                 predictive_forward=True, predictive_backward=True,
                 msb_bits=4, msb_bits_weight=4, msb_bits_grad=16,
                 threshold=5e-5, sparsify=False, sign=False,
                 writer=None, writer_prefix=""):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PredictiveConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)

        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.input_signed = input_signed
        self.predictive_forward = predictive_forward
        self.predictive_backward = predictive_backward
        self.msb_bits = msb_bits
        self.msb_bits_weight = msb_bits_weight
        self.msb_bits_grad = msb_bits_grad
        self.threshold = threshold
        self.sparsify = sparsify
        self.sign = sign
        self.writer = writer
        self.writer_prefix = writer_prefix
        self.counter = 0

        assert self.predictive_backward and self.msb_bits is not None

        self.quant_input = EfficientQuantize(
            num_bits=self.num_bits, msb_bits=self.msb_bits,
            shape_measure=(1,1,1,1,), flatten_dims=(1,-1), dequantize=True,
            input_signed=self.input_signed, stochastic=False, momentum=0.1)

        # if not self.predictive_backward:
        #     self.msb_bits_grad = None
        #     if not self.predictive_forward:
        #         self.msb_bits = self.msb_bits_weight = None
        #     else:
        #         assert self.msb_bits is not None and self.msb_bits_weight is not None
        # else:
        #     assert self.msb_bits is not None and self.msb_bits_weight is not None

    def forward(self, input):
        # Quantize `input` to `q_input`
        q_input, msb_input = self.quant_input(input)

        # Quantize `q_input` to get `msb_input`
        # if self.msb_bits is not None:
        #     msb_input = quantize(
        #         input.detach(), num_bits=self.msb_bits,
        #         flatten_dims=(1,-1), reduce_dim=0, signed=self.input_signed)
        # else:
        #     msb_input = None

        # Quantize weight
        q_weight, msb_weight = efficient_quant_weight(
            self.weight, num_bits_weight=self.num_bits_weight,
            msb_bits_weight=self.msb_bits_weight, threshold=self.threshold,
            sparsify=self.sparsify, sign=self.sign,
            writer=self.writer, writer_prefix=self.writer_prefix, counter=self.counter)
        # weights = quant_weight(
        #     self.weight, num_bits_weight=self.num_bits_weight,
        #     msb_bits_weight=self.msb_bits_weight, threshold=self.threshold,
        #     sparsify=self.sparsify, sign=self.sign,
        #     writer=self.writer, writer_prefix=self.writer_prefix, counter=self.counter)
        # q_weight = weights[0]
        # msb_weight = weights[1] if len(weights) > 1 else None
        self.counter += 1

        # No bias for CONV layers
        q_bias = None

        # Q-branch
        if not self.biprecision or self.num_bits_grad is None or self.num_bits_grad >= 32:
            q_output = F.conv2d(q_input, q_weight, bias=q_bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation, groups=self.groups)
            if self.num_bits_grad is not None and self.num_bits_grad < 32:
                q_output = quantize_grad(
                    q_output, num_bits=self.num_bits_grad, flatten_dims=(1,-1))
        else:
            q_output = conv2d_biprec(q_input, q_weight, q_bias, self.stride,
                                     self.padding, self.dilation, self.groups,
                                     num_bits_grad=self.num_bits_grad)

        # MSB-branch
        if msb_input is not None or msb_weight is not None:
            msb_output = F.conv2d(msb_input, msb_weight, bias=q_bias, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation, groups=self.groups)
            if self.predictive_backward:
                msb_output = quantize_grad(
                    msb_output, num_bits=self.msb_bits_grad, flatten_dims=(1,-1))
        else:
            msb_output = None

        # Mixing `q_output` and `msb_output`
        output = mixing_output(
            q_output, msb_output, self.predictive_forward, self.predictive_backward)

        return output
