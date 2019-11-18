from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['max_values', 'num_bits'])
EfficientQParams = namedtuple('QParams', ['max_values', 'num_bits', 'msb_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,
                      reduce_type='mean', keepdim=False):
    with torch.no_grad():
        x_flat_abs = x.abs().flatten(*flatten_dims)
        if x_flat_abs.dim() == 1:
            max_values = _deflatten_as(x_flat_abs.max(), x)
        else:
            max_values = _deflatten_as(x_flat_abs.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        return QParams(max_values=max_values, num_bits=num_bits)


def calculate_qparams_efficient(
    x, num_bits, msb_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,
    reduce_type='mean', keepdim=False):
    with torch.no_grad():
        x_flat_abs = x.abs().flatten(*flatten_dims)
        if x_flat_abs.dim() == 1:
            max_values = _deflatten_as(x_flat_abs.max(), x)
        else:
            max_values = _deflatten_as(x_flat_abs.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        return EfficientQParams(max_values=max_values, num_bits=num_bits, msb_bits=msb_bits)


class FPQuantizeFunction(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=32, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        # the following two `if` statements are just used to save computation
        if qparams is not None and qparams.num_bits >= 32:
            return input
        if qparams is None:
            assert num_bits is not None
            if num_bits >= 32:
                return input

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            qparams = calculate_qparams(output, num_bits=num_bits,
                                        flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        num_bits = qparams.num_bits
        max_values = qparams.max_values
        min_values = (- 1. * max_values) if signed else 0.
        delta = (max_values - min_values) / 2.**num_bits
        # delta = (max_values - min_values) / (2.**num_bits - 1)
        qmin, qmax = 0.0, 2.**num_bits - 1
        with torch.no_grad():
            output.sub_(min_values).div_(delta).clamp_(qmin,qmax).round_()

            if dequantize:
                output.mul_(delta).add_(min_values)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class EfficientFPQuantizeFunction(Function):

    @staticmethod
    def forward(ctx, input, num_bits=32, msb_bits=32, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        only_quantize_msb = False

        # the following two `if` statements are just used to save computation
        if qparams is not None and qparams.num_bits >= 32:
            q_input = input
            if qparams.msb_bits >= 32:
                msb_input = input.detach()
                return q_input, msb_input
            else:
                only_quantize_msb = True

        if qparams is not None:
            num_bits = qparams.num_bits
            msb_bits = qparams.msb_bits
        else:
            assert num_bits is not None and msb_bits is not None
            if num_bits >= 32:
                q_input = input
                if msb_bits >= 32:
                    msb_input = input.detach()
                else:
                    only_quantize_msb = True

        # ctx.inplace = inplace

        # if ctx.inplace:
        #     ctx.mark_dirty(input)
        #     output = input
        # else:
        #     output = input.clone()

        # ctx.mark_dirty(input)
        q_output = input.clone()
        # msb_output = input.detach()

        if qparams is None:
            qparams = calculate_qparams_efficient(
                q_output, num_bits=num_bits, msb_bits=msb_bits,
                flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        num_bits = qparams.num_bits
        msb_bits = qparams.msb_bits
        max_values = qparams.max_values
        min_values = (- 1. * max_values) if signed else 0.
        delta = (max_values - min_values) / 2.**num_bits
        # delta = (max_values - min_values) / (2.**num_bits - 1)
        qmin, qmax = 0.0, 2.**num_bits - 1
        scale = 2.0 ** (num_bits - msb_bits)
        with torch.no_grad():
            if not only_quantize_msb:
                q_output.sub_(min_values).div_(delta).clamp_(qmin,qmax).round_()
                msb_output = q_output.clone().div_(scale).round_().mul_(scale).clamp_(qmin, qmax)
            else:
                delta = (max_values - min_values) / 2.**msb_bits
                qmin, qmax = 0.0, 2.**msb_bits - 1
                msb_output = q_output.detach().sub_(min_values).div_(delta).clamp_(qmin,qmax).round_()

            if dequantize:
                if not only_quantize_msb:
                    q_output.mul_(delta).add_(min_values)
                msb_output.mul_(delta).add_(min_values)

        return q_output, msb_output

    @staticmethod
    def backward(ctx, *grad_output):
        # straight-through estimator
        grad_input = grad_output[0]
        return grad_input, None, None, None, None, None, None, None, None, None


class FPQuantizeGradFunction(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=32, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=True, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.dequantize = dequantize
        ctx.signed = signed
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.qparams is not None and ctx.qparams.num_bits >= 32:
            return grad_output

        if ctx.qparams is None:
            assert ctx.num_bits is not None
            if ctx.num_bits >= 32:
                return grad_output

        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits,
                    flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                    reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None, None


def quantize(x, num_bits=None, qparams=None,
             flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,
             dequantize=True, signed=False, stochastic=False, inplace=False):
    return FPQuantizeFunction().apply(
        x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def efficient_quantize(x, num_bits=None, msb_bits=None, qparams=None,
                       flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,
                       dequantize=True, signed=False, stochastic=False, inplace=False):
    return EfficientFPQuantizeFunction().apply(
        x, num_bits, msb_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)


def quantize_grad(x, num_bits=None, qparams=None,
                  flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0,
                  dequantize=True, signed=True, stochastic=False):
    return FPQuantizeGradFunction().apply(
        x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic)


class Quantize(nn.Module):
    """docstring for Quantize"""

    def __init__(self, num_bits=8, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 dequantize=True, input_signed=False, stochastic=False, momentum=0.1):
        super(Quantize, self).__init__()
        self.register_buffer('running_max_values', torch.zeros(*shape_measure))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.input_signed = input_signed
        self.stochastic = stochastic
        self.num_bits = num_bits

    def forward(self, input, qparams=None):

        # Quantize input
        if self.num_bits is not None and self.num_bits < 32:
            if self.training:
                if qparams is None:
                    qparams = calculate_qparams(
                        input, num_bits=self.num_bits, flatten_dims=self.flatten_dims, reduce_dim=0)
                with torch.no_grad():
                    self.running_max_values.mul_(self.momentum).add_(
                        qparams.max_values * (1 - self.momentum))
            else:
                qparams = QParams(max_values=self.running_max_values,
                                  num_bits=self.num_bits)
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               signed=self.input_signed, stochastic=self.stochastic, inplace=False)
        else:
            q_input = input

        return q_input


class EfficientQuantize(nn.Module):
    """docstring for Quantize"""

    def __init__(self, num_bits=8, msb_bits=4, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 dequantize=True, input_signed=False, stochastic=False, momentum=0.1):
        super(EfficientQuantize, self).__init__()
        self.register_buffer('running_max_values', torch.zeros(*shape_measure))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.input_signed = input_signed
        self.stochastic = stochastic
        self.num_bits = num_bits
        self.msb_bits = msb_bits

    def forward(self, input, qparams=None):

        # Quantize input
        # if self.num_bits is not None:
        if self.training:
            if qparams is None:
                qparams = calculate_qparams_efficient(
                    input, num_bits=self.num_bits, msb_bits=self.msb_bits,
                    flatten_dims=self.flatten_dims, reduce_dim=0)
            with torch.no_grad():
                self.running_max_values.mul_(self.momentum).add_(
                    qparams.max_values * (1 - self.momentum))
        else:
            qparams = EfficientQParams(
                max_values=self.running_max_values, num_bits=self.num_bits, msb_bits=self.msb_bits)
        q_input, msb_input = efficient_quantize(
            input, qparams=qparams, dequantize=self.dequantize,
            signed=self.input_signed, stochastic=self.stochastic, inplace=False)
        # else:
            # q_input = input

        return q_input, msb_input


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
