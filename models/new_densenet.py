import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch.autograd as autograd
from torch.autograd import  Variable
import numpy as np
import scipy.misc

from models.conv import PredictiveConv2d

NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = None,

BIPRECISION = False
PREDICTIVE_FORWARD = False
PREDICTIVE_BACKWARD = True
MSB_BITS = 4
MSB_BITS_WEIGHT = 4
MSB_BITS_GRAD = 16

THRESHOLD = 5e-5
SPARSIFY = False
SIGN = True
WRITER = None

__all__ = ['DenseNet', 'new_densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def conv1x1(in_planes, out_planes, stride=1,
            input_signed=True, predictive_forward=True, writer_prefix=""):
    "1x1 convolution with no padding"
    predictive_forward = PREDICTIVE_FORWARD and predictive_forward
    return PredictiveConv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
        num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD,
        biprecision=BIPRECISION, input_signed=input_signed,
        predictive_forward=predictive_forward, predictive_backward=PREDICTIVE_BACKWARD,
        msb_bits=MSB_BITS, msb_bits_weight=MSB_BITS_WEIGHT, msb_bits_grad=MSB_BITS_GRAD,
        threshold=THRESHOLD, sparsify=SPARSIFY, sign=SIGN,
        writer=WRITER, writer_prefix=writer_prefix)


def conv3x3(in_planes, out_planes, stride=1,
            input_signed=False, predictive_forward=True, writer_prefix=""):
    "3x3 convolution with padding"
    predictive_forward = PREDICTIVE_FORWARD and predictive_forward
    return PredictiveConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
        num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD,
        biprecision=BIPRECISION, input_signed=input_signed,
        predictive_forward=predictive_forward, predictive_backward=PREDICTIVE_BACKWARD,
        msb_bits=MSB_BITS, msb_bits_weight=MSB_BITS_WEIGHT, msb_bits_grad=MSB_BITS_GRAD,
        threshold=THRESHOLD, sparsify=SPARSIFY, sign=SIGN,
        writer=WRITER, writer_prefix=writer_prefix)

def new_densenet121(pretrained=False, **kwargs):
    global NUM_BITS
    global NUM_BITS_WEIGHT
    global NUM_BITS_GRAD
    global BIPRECISION
    global PREDICTIVE_FORWARD
    global PREDICTIVE_BACKWARD
    global MSB_BITS
    global MSB_BITS_WEIGHT
    global MSB_BITS_GRAD
    global THRESHOLD
    global SPARSIFY
    global SIGN
    global WRITER

    print('num_bits:', kwargs['num_bits'])
    print('num_bits_weight:', kwargs['num_bits_weight'])
    print('num_bits_grad:', kwargs['num_bits_grad'])
    print('biprecision:', kwargs['biprecision'])
    print('predictive_forward:', kwargs['predictive_forward'])
    print('predictive_backward:', kwargs['predictive_backward'])
    print('msb_bits:', kwargs['msb_bits'])
    print('msb_bits_weight:', kwargs['msb_bits_weight'])
    print('msb_bits_grad:', kwargs['msb_bits_grad'])
    print('threshold:', kwargs['threshold'])
    print('sparsify:', kwargs['sparsify'])
    print('sign:', kwargs['sign'])

    NUM_BITS = kwargs.pop('num_bits', 8)
    NUM_BITS_WEIGHT = kwargs.pop('num_bits_weight', 8)
    NUM_BITS_GRAD = kwargs.pop('num_bits_grad', None)
    BIPRECISION = kwargs.pop('biprecision', False)
    PREDICTIVE_FORWARD = kwargs.pop('predictive_forward', False)
    PREDICTIVE_BACKWARD = kwargs.pop('predictive_backward', True)
    MSB_BITS = kwargs.pop('msb_bits', 4)
    MSB_BITS_WEIGHT = kwargs.pop('msb_bits_weight', 4)
    MSB_BITS_GRAD = kwargs.pop('msb_bits_grad', 16)
    THRESHOLD = kwargs.pop('threshold', 5e-4)
    SPARSIFY = kwargs.pop('sparsify', False)
    SIGN = kwargs.pop('sign', True)
    WRITER = None

    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=24, growth_rate=12, block_config=(6, 12, 24, 16),
                     )
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet100(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=24, growth_rate=12, block_config=(16, 16, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model




def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.num_input_features = num_input_features
        self.num_output_features = growth_rate
        self.dense_module = nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(num_input_features)),
                      ('relu1', nn.ReLU()),
                      ('conv1', conv1x1(num_input_features, bn_size *
                        growth_rate, stride=1, input_signed=False, predictive_forward=False)),
                      ('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
                      ('relu2', nn.ReLU()),
                      ('conv2', conv3x3(bn_size * growth_rate, growth_rate,
                        input_signed=False, predictive_forward=False))])
                       )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.dense_module(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.trans_module = nn.Sequential(OrderedDict([('norm', nn.BatchNorm2d(num_input_features)),
                       ('relu', nn.ReLU()),
                       ('conv', conv1x1(num_input_features, num_output_features,
                                          stride=1, input_signed=False, predictive_forward=False)),
                       ('pool', nn.AvgPool2d(kernel_size=2, stride=2))]))
    def forward(self, x):
        return self.trans_module(x)






class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(16, 16, 16),
                 num_init_features=24, bn_size=4, drop_rate=0, num_classes=1000, embed_dim = 10, hidden_dim = 10):

        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        # First convolution

        self.base_layer = conv3x3(3, num_init_features, input_signed=False, predictive_forward=False)

        # self.base_layer = nn.Sequential(OrderedDict([
            # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
           # ('norm0', nn.BatchNorm2d(num_init_features)),
           # ('relu0', nn.ReLU(inplace=True)),
           # ('pool0', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
        # ]))
        self.block_config = block_config
        num_features = num_init_features

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.avg_pool_one = nn.AvgPool2d(kernel_size=2, stride=2)

        self.avg_pool_two = nn.AvgPool2d(kernel_size=2, stride=2)

        # self.control = RNNGate(embed_dim, hidden_dim, rnn_type = 'lstm')


        # denseblock 0
        for i in range(block_config[0]):
            setattr(self, 'denseblock0_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))

            input_channels = num_features + i * growth_rate


            gate_layer = nn.Sequential(
                nn.AvgPool2d(32),
                conv1x1(in_planes = num_features + (i + 1) * growth_rate,
                          out_planes = self.embed_dim,
                          stride = 1,
                          input_signed=False,
                          predictive_forward=False))


            setattr(self, 'denseblock0_%s_gate' % i, gate_layer)

            if input_channels != growth_rate:
                downsample = nn.Sequential(
                    conv1x1(input_channels,growth_rate,
                              stride = 1, input_signed=False, predictive_forward=False),
                    nn.BatchNorm2d(growth_rate),

                )

                setattr(self, 'denseblock0_%s_ds' % i, downsample)



        num_features = (num_features + block_config[0] * growth_rate)
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


        # denseblock 1
        for i in range(block_config[1]):
            setattr(self, 'denseblock1_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))

            input_channels = num_features + i * growth_rate

            gate_layer = nn.Sequential(
                nn.AvgPool2d(16),
                conv1x1(in_planes = num_features + (i + 1) * growth_rate,
                          out_planes = self.embed_dim,
                          stride = 1,
                          input_signed=False,
                          predictive_forward=False))


            setattr(self, 'denseblock1_%s_gate' % i, gate_layer)


            if input_channels != growth_rate:
                downsample = nn.Sequential(
                    conv1x1(input_channels,growth_rate,
                              stride = 1, input_signed=False, predictive_forward=False),
                    nn.BatchNorm2d(growth_rate),

                )

                setattr(self, 'denseblock1_%s_ds' % i, downsample)



        num_features = (num_features + block_config[1] * growth_rate)
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2



        # denseblock 2
        for i in range(block_config[2]):
            setattr(self, 'denseblock2_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))

            input_channels = num_features + i * growth_rate

            gate_layer = nn.Sequential(
                nn.AvgPool2d(8),
                conv1x1(in_planes = num_features + (i + 1) * growth_rate,
                          out_planes = self.embed_dim,
                          stride = 1,
                          input_signed=False, predictive_forward=False))


            setattr(self, 'denseblock2_%s_gate' % i, gate_layer)

            if input_channels != growth_rate:
                downsample = nn.Sequential(
                    conv1x1(input_channels,growth_rate,
                              stride = 1, input_signed = False, predictive_forward=False),
                    nn.BatchNorm2d(growth_rate),

                )

                setattr(self, 'denseblock2_%s_ds' % i, downsample)


        num_features = (num_features + block_config[2] * growth_rate)
       # self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
       # num_features = num_features // 2


        # denseblock 3
       # for i in range(block_config[3]):
           # setattr(self, 'denseblock3_%s' % i, self._make_layer(i, i+1, num_features, growth_rate, bn_size, drop_rate))
       # num_features = (num_features + block_config[3] * growth_rate)


        # Final batch norm
        self.bn_norm = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)



        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, front_layer_idx, back_layer_index, num_input_features, growth_rate, bn_size, drop_rate):
        modules = []
        for i in range(front_layer_idx, back_layer_index):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            modules.extend([layer])
        return nn.Sequential(*modules)

    def install_gate(self):
        self.control = RNNGate(self.embed_dim, self.hidden_dim, rnn_type='lstm')


    def forward(self, x):

        batch_size = x.size(0)
        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        features = self.base_layer(x)

        masks = []
        gprobs = []

        new_features = getattr(self, 'denseblock0_0')(features)

        prev_new_features = new_features

        features = torch.cat([features, new_features], 1)

        gate_feature = getattr(self, 'denseblock0_0_gate')(features)
        mask,gprob = self.control(gate_feature)

        gprobs.append(gprob)
        masks.append(mask.squeeze())

        prev = features

        # loop for denseblock 0
        for i in range(1, self.block_config[0]):

            # if getattr(self, 'denseblock0_{}_ds'.format(i)) is not None:
                # prev = getattr(self, 'denseblock0_{}_ds'.format(i))(prev)

            new_features = getattr(self, 'denseblock0_{}'.format(i))(features)
            new_features = mask.expand_as(new_features) * new_features + (1 - mask).expand_as(prev_new_features) * prev_new_features

            prev_new_features = new_features

            features = torch.cat([features, new_features], 1)

            gate_feature = getattr(self, 'denseblock0_{}_gate'.format(i))(features)
            mask, gprob = self.control(gate_feature)

            gprobs.append(gprob)
            masks.append(mask.squeeze())

            prev = features



        features = self.trans0(features)
        prev = features

        # new_features = getattr(self, 'denseblock1_0')(features)

        # prev_new_features = new_features

        # features = torch.cat([features, new_features], 1)

        # gate_feature = getattr(self, 'denseblock1_0_gate')(features)
        # mask,gprob = self.control(gate_feature)

        # gprobs.append(gprob)
        # masks.append(mask.squeeze())

        # prev = features


        # loop for denseblock 1
        for i in range(self.block_config[1]):

            if i == 0:
                prev_new_features = self.avg_pool_one(prev_new_features)

            # if getattr(self, 'denseblock1_{}_ds'.format(i)) is not None:
                # prev = getattr(self, 'denseblock1_{}_ds'.format(i))(prev)

            new_features = getattr(self, 'denseblock1_{}'.format(i))(features)

            new_features = mask.expand_as(new_features) * new_features + (1 - mask).expand_as(prev_new_features) * prev_new_features

            prev_new_features = new_features

            features = torch.cat([features, new_features], 1)

            # print('group_one')
            # print(i)

            gate_feature = getattr(self, 'denseblock1_{}_gate'.format(i))(features)
            mask, gprob = self.control(gate_feature)

            gprobs.append(gprob)
            masks.append(mask.squeeze())

            prev = features

        features = self.trans1(features)
        prev = features

        # new_features = getattr(self, 'denseblock2_0')(features)

        # prev_new_features = new_features

        # features = torch.cat([features, new_features], 1)

        # gate_feature = getattr(self, 'denseblock2_0_gate')(features)
        # mask,gprob = self.control(gate_feature)

        # gprobs.append(gprob)
        # vmasks.append(mask.squeeze())

        # vprev = features

        # loop for denseblock 2
        for i in range(self.block_config[2]):

            # print('group_two')
            # print(i)

            # if getattr(self, 'denseblock2_{}_ds'.format(i)) is not None:
                # prev = getattr(self, 'denseblock2_{}_ds'.format(i))(prev)

            if i == 0:
                prev_new_features = self.avg_pool_two(prev_new_features)

            new_features = getattr(self, 'denseblock2_{}'.format(i))(features)
            new_features = mask.expand_as(new_features) * new_features + (1 - mask).expand_as(prev_new_features) * prev_new_features

            prev_new_features = new_features

            features = torch.cat([features, new_features], 1)


            gate_feature = getattr(self, 'denseblock2_{}_gate'.format(i))(features)
            mask, gprob = self.control(gate_feature)

            if i < self.block_config[2] - 1:

                gprobs.append(gprob)
                masks.append(mask.squeeze())

            prev = features
       # features = self.trans2(features)

        # loop for denseblock 3
       # for i in range(self.block_config[3]):
           # new_features = getattr(self, 'denseblock3_{}'.format(i))(features)
           # features = torch.cat([features, new_features], 1)

        features = self.bn_norm(features)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)


        return out, masks, gprobs






########################################
# DenseNet with Feedforward Gate     #
########################################
# FFGate-II
class FeedforwardGateII(nn.Module):
    """ use single conv (stride=2) layer only"""
    def __init__(self, pool_size=5, channel=10):
        super(FeedforwardGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel
        self.activate = False
        self.energy_cost = 0
        self.conv1 = conv3x3(channel, channel, stride=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # pool_size = math.floor(pool_size/2 + 0.5) # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)

        x = self.linear_layer(x)

        print(x.size())

        x = x.view(x.size(0), -1)

        softmax = self.prob_layer(x)
        logprob = self.logprob(x)
        # discretize
        x = (softmax[:, 1] > 0.4).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]

        x = x.view(x.size(0), 1, 1, 1)
        return x, logprob




# ======================
# Recurrent Gate  Design
# ======================

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(nn.Module):
    """given the fixed input size, return a single layer lstm """
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm',output_channel=1):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        self.proj = nn.Linear(hidden_dim, output_channel)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        batch_size = x.size(0)
        self.rnn.flatten_parameters()

        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        out = out.squeeze()

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
        disc_prob = disc_prob.view(batch_size, 1, 1, 1)

        return disc_prob, prob


