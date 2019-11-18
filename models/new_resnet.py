import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd

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

WRITER_PREFIX_COUNTER = 0


def conv1x1(in_planes, out_planes, stride=1, input_signed=True, predictive_forward=True, writer_prefix=""):
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


def conv3x3(in_planes, out_planes, stride=1, input_signed=False, predictive_forward=True, writer_prefix=""):
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, writer_prefix=""):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, input_signed=True,
                             predictive_forward=False, writer_prefix=writer_prefix+'_conv1')
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, input_signed=False,
                             predictive_forward=False, writer_prefix=writer_prefix+'_conv2')
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# For Recurrent Gate
def repackage_hidden(h):
    """ to reduce memory usage"""
    if h is None:
        return None
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(nn.Module):
    """Recurrent Gate definition.
    Input is already passed through average pooling and embedding."""
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm', output_channel=1):
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
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda(), requires_grad=True),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda(), requires_grad=True))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        # Take the convolution output of each step
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        # prob = nn.functional.relu(prob - 0.1)

        tmp = torch.rand_like(prob)
        disc_prob = (prob > tmp).float().detach() - \
                    prob.detach() + prob

        disc_prob = disc_prob.view(batch_size, -1, 1, 1)
        return disc_prob, prob


class ResNetRecurrentGateSP(nn.Module):
    """SkipNet with Recurrent Gate Model"""
    def __init__(self, block, layers, num_classes=10, embed_dim=10, hidden_dim=10):
        self.inplanes = 16
        super(ResNetRecurrentGateSP, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16, input_signed=True, predictive_forward=False, writer_prefix='conv1')
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self._make_group(block, 16, layers[0], group_id=1, pool_size=32, writer_prefix='group_1')
        self._make_group(block, 32, layers[1], group_id=2, pool_size=16, writer_prefix='group_2')
        self._make_group(block, 64, layers[2], group_id=3, pool_size=8, writer_prefix='group_3')

        # define recurrent gating module
        self.avgpool = nn.AvgPool2d(8)
        print(num_classes)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, PredictiveConv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def install_gate(self):
        self.control = RNNGate(self.embed_dim, self.hidden_dim, rnn_type='lstm', output_channel=1)

    def _make_group(self, block, planes, layers, group_id=1, pool_size=16, writer_prefix=''):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size, writer_prefix=writer_prefix+'_layer%d'%i)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1, pool_size=16, writer_prefix=''):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride,
                        input_signed=True, predictive_forward=False, writer_prefix=writer_prefix+'_downsample'),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = block(self.inplanes, planes, stride, downsample, writer_prefix=writer_prefix)

        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            conv1x1(planes * block.expansion, self.embed_dim, stride=stride,
                    input_signed=True, predictive_forward=False, writer_prefix=writer_prefix+'_gate')
        )
        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def forward(self, x):

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer

        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        for g in range(3):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)

                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                # new mask is taking the current output
                prev = x = mask.expand_as(x) * x \
                           + (1 - mask).expand_as(prev) * prev

                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                # control = getattr(self, 'control{}'.format(min(3, g + 1 + (i == self.num_layers[g] - 1))))
                mask, gprob = self.control(gate_feature)
                gprobs.append(gprob)
                masks.append(mask.squeeze())

        # last block doesn't have gate module
        del masks[-1]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks, gprobs


# For CIFAR-10
def cifar10_rnn_gate_74(**kwargs):
    """SkipNet-74 with Recurrent Gate"""

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
    print('writer:', kwargs['writer'])

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
    WRITER = kwargs.pop('writer', None)

    model = ResNetRecurrentGateSP(BasicBlock, [12, 12, 12], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_110(**kwargs):
    """SkipNet-110 with Recurrent Gate"""

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
    print('writer:', kwargs['writer'])

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
    WRITER = kwargs.pop('writer', None)

    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model

# For CIFAR-100


def cifar100_rnn_gate_110(**kwargs):
    """SkipNet-110 with Recurrent Gate """

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
    print('writer:', kwargs['writer'])

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
    WRITER = kwargs.pop('writer', None)

    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


