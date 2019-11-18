import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d
from torch.nn.modules.utils import _pair
from torch.autograd import Function

from models.quantize import calculate_qparams, quantize, quantize_grad
from models.quantize import efficient_quantize


# Inherit from Function
class PredictiveForwardMixingFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, q_out, msb_out, predictive_forward, predictive_backward): # , msb_bits_grad):
        ctx.save_for_backward(msb_out)
        ctx.predictive_backward = predictive_backward

        if msb_out is None or not predictive_forward:
            return q_out
        else:
            with torch.no_grad():
                nne_locs = (msb_out >= 0).detach().float()
                return q_out * nne_locs

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        msb_out = ctx.saved_tensors[0]
        grad_q_out = grad_output
        grad_msb_out = None
        with torch.no_grad():
            if msb_out is not None and ctx.predictive_backward:
                grad_msb_out = grad_output.clone()

        return grad_q_out, grad_msb_out, None, None


class PredictiveWeightQuantFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, weight, num_bits_weight, msb_bits_weight,
                threshold, sparsify, sign, writer, writer_prefix, counter):
        ctx.threshold = threshold
        ctx.sparsify = sparsify
        ctx.sign = sign
        ctx.writer = writer
        ctx.writer_prefix = writer_prefix
        ctx.counter = counter

        with torch.no_grad():
            # q_weight
            if num_bits_weight is not None and num_bits_weight < 32:
                q_weight = quantize(
                    weight, num_bits=num_bits_weight,
                    flatten_dims=(1,-1), reduce_dim=None, signed=True)
            else:
                q_weight = weight

            # msb_weight
            if msb_bits_weight is None:
                return (q_weight,)
            elif msb_bits_weight < 32:
                msb_weight = quantize(
                    weight, num_bits=msb_bits_weight,
                    flatten_dims=(1,-1), reduce_dim=None, signed=True)
            else:
                msb_weight = weight.clone()

            return q_weight, msb_weight

    # This function has two outputs, so it gets two gradients
    @staticmethod
    def backward(ctx, *grad_output):
        grad_weight = None
        writer = ctx.writer
        prefix = ctx.writer_prefix
        counter = ctx.counter
        grad_q_weight = grad_output[0]
        grad_msb_weight = grad_output[1] if len(grad_output) > 1 else None
        # if writer is not None and counter % 400 == 0:
            # writer.add_scalar(prefix+'/grad_q_weight_max', grad_q_weight.abs().max(), counter)

        with torch.no_grad():
            if grad_msb_weight is not None:
                grad_msb_weight_abs = grad_msb_weight.abs()
                if ctx.threshold < 0:
                    ctx.threshold = -1.0 * ctx.threshold * grad_msb_weight_abs.max()
                large_locs = (grad_msb_weight_abs >= ctx.threshold).detach().float()
                if ctx.sparsify:
                    grad_weight = large_locs * grad_msb_weight
                else:
                    grad_weight = large_locs * grad_msb_weight + (1 - large_locs) * grad_q_weight

                grad_msb_sign_correct_locs = (grad_msb_weight.sign() == grad_q_weight.sign()).float()
                grad_msb_sign_wrong = grad_msb_weight * (1 - grad_msb_sign_correct_locs)
                if writer is not None and counter % 100 == 0:
                    writer.add_scalar(prefix+'/ratio_grad_msb_used', float(large_locs.sum()) / float(large_locs.numel()), counter)
                    writer.add_scalar(prefix+'/grad_numel', float(large_locs.numel()), counter)
            else:
                grad_weight = grad_q_weight

            if ctx.sign:
                grad_weight.sign_()
                # grad_weight = quantize(grad_weight, num_bits=you_set,
                #                        flatten_dims=(1,-1), signed=True)

            return grad_weight, None, None, None, None, None, None, None, None


class EfficientPredictiveWeightQuantFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, weight, num_bits_weight, msb_bits_weight,
                threshold, sparsify, sign, writer, writer_prefix, counter):
        ctx.threshold = threshold
        ctx.sparsify = sparsify
        ctx.sign = sign
        ctx.writer = writer
        ctx.writer_prefix = writer_prefix
        ctx.counter = counter

        with torch.no_grad():
            # q_weight
            if ((num_bits_weight is None or num_bits_weight >= 32) and
                (msb_bits_weight is None or msb_bits_weight >= 32)):
                return weight, weight.detach()
            else:
                q_weight, msb_weight = efficient_quantize(
                    weight, num_bits=num_bits_weight, msb_bits=msb_bits_weight,
                    flatten_dims=(1,-1), reduce_dim=None, signed=True)
                return q_weight, msb_weight

    # This function has two outputs, so it gets two gradients
    @staticmethod
    def backward(ctx, *grad_output):
        grad_weight = None
        writer = ctx.writer
        prefix = ctx.writer_prefix
        counter = ctx.counter
        grad_q_weight = grad_output[0]
        grad_msb_weight = grad_output[1] # if len(grad_output) > 1 else None

        with torch.no_grad():
            if grad_msb_weight is not None:
                grad_msb_weight_abs = grad_msb_weight.abs()
                if ctx.threshold < 0:
                    ctx.threshold = -1.0 * ctx.threshold * grad_msb_weight_abs.max()
                large_locs = (grad_msb_weight_abs >= ctx.threshold).detach().float()
                if ctx.sparsify:
                    grad_weight = large_locs * grad_msb_weight
                else:
                    grad_weight = large_locs * grad_msb_weight + (1 - large_locs) * grad_q_weight

                if writer is not None and counter % 200 == 0:
                    grad_msb_sign_correct_locs = (grad_msb_weight.sign() == grad_q_weight.sign()).float()
                    grad_msb_sign_wrong = grad_msb_weight * (1 - grad_msb_sign_correct_locs)
                    writer.add_scalar(prefix+'/ratio_grad_msb_used', float(large_locs.sum()) / float(large_locs.numel()), counter)
                    writer.add_scalar(prefix+'/grad_numel', float(large_locs.numel()), counter)
            else:
                grad_weight = grad_q_weight

            if ctx.sign:
                grad_weight.sign_()

            return grad_weight, None, None, None, None, None, None, None, None


class PredictiveBiasQuantFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, bias, num_bits_bias, msb_bits_bias, threshold, sparsify, sign):
        ctx.threshold = threshold
        ctx.sparsify = sparsify
        ctx.sign = sign

        with torch.no_grad():
            # q_weight
            if num_bits_bias is not None and num_bits_bias < 32:
                q_bias = quantize(bias, num_bits=num_bits_bias,
                                  flatten_dims=(0,-1), reduce_dim=0, signed=True)
            else:
                q_bias = bias

            # msb_weight
            if msb_bits_bias is None:
                return (q_bias,)
            elif msb_bits_bias < 32:
                msb_bias = quantize(bias, num_bits=msb_bits_bias,
                                    flatten_dims=(0,-1), reduce_dim=0, signed=True)
            else:
                msb_bias = bias.clone()

            return q_bias, msb_bias

    # This function has two outputs, so it gets two gradients
    @staticmethod
    def backward(ctx, *grad_output):
        grad_bias = None
        grad_q_bias = grad_output[0]
        grad_msb_bias = grad_output[1] if len(grad_output) > 1 else None

        with torch.no_grad():
            if grad_msb_bias is not None:
                large_locs = (grad_msb_bias.abs() >= ctx.threshold).detach().float()
                if ctx.sparsify:
                    grad_bias = large_locs * grad_msb_bias
                else:
                    grad_bias = large_locs * grad_msb_bias + (1 - large_locs) * grad_q_bias
            else:
                grad_bias = grad_q_bias.clone()

            if ctx.sign:
                grad_bias.sign_()

            return grad_bias, None, None, None, None, None


def mixing_output(q_out, msb_out, predictive_forward, predictive_backward): # , msb_bits_grad=16):
    return PredictiveForwardMixingFunction.apply(
        q_out, msb_out, predictive_forward, predictive_backward) # , msb_bits_grad)


def quant_weight(weight, num_bits_weight=8, msb_bits_weight=4,
                 threshold=5e-4, sparsify=False, sign=False,
                 writer=None, writer_prefix="", counter=0):
    return PredictiveWeightQuantFunction.apply(
        weight, num_bits_weight, msb_bits_weight, threshold, sparsify, sign,
        writer, writer_prefix, counter)


def efficient_quant_weight(weight, num_bits_weight=8, msb_bits_weight=4,
                           threshold=5e-4, sparsify=False, sign=False,
                           writer=None, writer_prefix="", counter=0):
    return EfficientPredictiveWeightQuantFunction.apply(
        weight, num_bits_weight, msb_bits_weight, threshold, sparsify, sign,
        writer, writer_prefix, counter)


def quant_bias(weight, num_bits_bias=16, msb_bits_bias=8,
               threshold=5e-4, sparsify=False, sign=False):
    return PredictiveBiasQuantFunction.apply(
        weight, num_bits_bias, msb_bits_bias, threshold, sparsify, sign)
