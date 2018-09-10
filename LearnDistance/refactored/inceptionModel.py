import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
import torch.utils.model_zoo as model_zoo
from helperFunctions import model_weights_random_xavier

__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def featsInception(outDim=3, pretrained=False, **kwargs):
    model = featsInceptionCifar10(outDim=outDim)
    model_weights_random_xavier(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
    return model

def distInception(pretrained=False):
    model = distInceptionCifar10()
    model_weights_random_xavier(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
    return model


def featsInceptionAE(outDim=3, pretrained=False, **kwargs):
    model = featsInceptionCifar10AE(outDim=outDim)
    model_weights_random_xavier(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
    return model

class featsInceptionCifar10(nn.Module):

    def __init__(self, outDim=3):
        super(featsInceptionCifar10, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ('c1', BasicConv2d(3, 96, kernel_size=3, stride=1)),
            ('i1a', InceptionModule(96, 32, 32)),
            ('i1b', InceptionModule(64, 32, 48)),
            ('d1', DownsampleModule(80, 80)),
            ('i2a', InceptionModule(160, 112, 48)),
            ('i2b', InceptionModule(160, 96, 64)),
            ('i2c', InceptionModule(160, 80, 80)),
            ('i2d', InceptionModule(160, 48, 96)),
            ('d2', DownsampleModule(144, 96)),
            ('i3a', InceptionModule(240, 176, 160)),
            ('i3b', InceptionModule(336, 176, 160)),
            ('mp', nn.AvgPool2d(7)),
        ]))

        self.fc = nn.Linear(336, outDim)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 336)
        output = self.fc(output)
        return output


class distInceptionCifar10(nn.Module):

    def __init__(self):
        super(distInceptionCifar10, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('c1', BasicConv2d(6, 96, kernel_size=3, stride=1)),
            ('i1a', InceptionModule(96, 32, 32)),
            ('i1b', InceptionModule(64, 32, 48)),
            ('d1', DownsampleModule(80, 80)),
            ('i2a', InceptionModule(160, 112, 48)),
            ('i2b', InceptionModule(160, 96, 64)),
            ('i2c', InceptionModule(160, 80, 80)),
            ('i2d', InceptionModule(160, 48, 96)),
            ('d2', DownsampleModule(144, 96)),
            ('i3a', InceptionModule(240, 176, 160)),
            ('i3b', InceptionModule(336, 176, 160)),
            ('mp', nn.AvgPool2d(7)),
        ]))

        self.fc = nn.Linear(336, 1)

    def forward(self, input1, input2):
        input = cat((input1,input2),1)
        output = self.net(input)
        output = output.view(-1, 336)
        output = self.fc(output)
        return output


class featsInceptionCifar10AE(nn.Module):

    def __init__(self, outDim=3):
        super(featsInceptionCifar10AE, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ('c1', BasicConv2d(3, 96, kernel_size=3, stride=1)),
            ('i1a', InceptionModule(96, 32, 32)),
            ('i1b', InceptionModule(64, 32, 48)),
            ('d1', DownsampleModule(80, 80)),
            ('i2a', InceptionModule(160, 112, 48)),
            ('i2b', InceptionModule(160, 96, 64)),
            ('i2c', InceptionModule(160, 80, 80)),
            ('i2d', InceptionModule(160, 48, 96)),
            ('d2', DownsampleModule(144, 96)),
            ('i3a', InceptionModule(240, 176, 160)),
            ('i3b', InceptionModule(336, 176, 160)),
            ('mp', nn.AvgPool2d(7)),
        ]))

        self.fc = nn.Linear(336, outDim)

        self.decoderFc = nn.Linear(outDim, 336)

        self.decoderNet = nn.Sequential(OrderedDict([
            ('mprev', nn.ConvTranspose2d(7)),
            ('i3brev', InceptionModuleRev(336, 176, 160)),
            ('i3arev', InceptionModuleRev(336, 144, 96)),
            ('d2rev', DownsampleModuleRev(240, 144)),
            ('i2drev', InceptionModuleRev(144, 80, 80)),
            ('i2crev', InceptionModuleRev(160, 96, 64)),
            ('i2brev', InceptionModuleRev(160, 112, 48)),
            ('i2arev', InceptionModuleRev(160, 80, 80)),
            ('d1rev', DownsampleModuleRev(160, 80)),
            ('i1brev', InceptionModuleRev(80, 32, 48)),
            ('i1arev', InceptionModuleRev(80, 32, 32)),
            ('c1rev', BasicConv2dRev(64, 3, kernel_size=3, stride=1)),
        ]))

    def encoder(self, input):
        output = self.net(input)
        output = output.view(-1, 336)
        output = self.fc(output)
        return output

    def decoder(self, input):
        output = self.decoderFc(input)
        output = output.view(-1, 336, 1, 1)
        output = self.decoderNet(output)
        return output

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out1_channels, out2_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, out1_channels, kernel_size=1)
        self.branch3x3 = BasicConv2d(in_channels, out2_channels, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)

        outputs = [branch1x1, branch3x3]
        return torch.cat(outputs, 1)


class InceptionModuleRev(nn.Module):

    def __init__(self, in_channels, out1_channels, out2_channels):
        super(InceptionModuleRev, self).__init__()
        self.branch1x1 = BasicConv2dRev(in_channels, out1_channels, kernel_size=1)
        self.branch3x3 = BasicConv2dRev(in_channels, out2_channels, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)

        outputs = [branch1x1, branch3x3]
        return torch.cat(outputs, 1)


class DownsampleModule(nn.Module):

    def __init__(self, in_channels, conv_channels):
        super(DownsampleModule, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, conv_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        outputs = [branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class DownsampleModuleRev(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super(DownsampleModuleRev, self).__init__()
        self.branch3x3 = BasicConv2dRev(in_channels, conv_channels, kernel_size=3, stride=2, padding=1)
        # self.branchUpPool = BasicConv2dRev(in_channels, conv_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        # branch_pool = F.conv_transpose2d(x, kernel_size=3, stride=2, padding=1)

        outputs = [branch3x3]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)



class BasicConv2dRev(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2dRev, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)