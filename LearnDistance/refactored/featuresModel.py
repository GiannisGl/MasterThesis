from collections import OrderedDict
import torch
import torch.nn as nn
from helperFunctions import model_weights_random_xavier, initialize_pretrained_model
import sys
sys.path.insert(0, '../../trainModels')

class FeatsLeNet5(nn.Module):

    # Input - 1x28x28
    # C1 - 6@24x24 (5x5 kernel)
    # ELU
    # S2 - 6@12x12 (2x2 kernel, stride 2) Subsampling
    # C3 - 16@8x8 (5x5 kernel)
    # ELU
    # S4 - 16@4x4 (2x2 kernel, stride 2) Subsampling
    # C5 - 120@1x1 (5x5 kernel)
    # F6 - 84
    # ELU
    # F7 - outDim (Output)

    def __init__(self, outDim):
        super(FeatsLeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ELU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ELU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ELU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ELU()),
            ('f7b', nn.Linear(84, outDim)),
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120)
        output = self.fc(output)
        return output



class FeatsLeNet5Orig(nn.Module):

    # Input - 1x28x28
    # C1 - 6@24x24 (5x5 kernel)
    # ELU
    # S2 - 6@12x12 (2x2 kernel, stride 2) Subsampling
    # C3 - 16@8x8 (5x5 kernel)
    # ELU
    # S4 - 16@4x4 (2x2 kernel, stride 2) Subsampling
    # C5 - 120@1x1 (5x5 kernel)
    # F6 - 84
    # ELU
    # F7 - 10 (Output)
    # ELU

    def __init__(self):
        super(FeatsLeNet5Orig, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ELU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ELU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ELU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ELU()),
            ('f7', nn.Linear(84, 10)),
            ('relu7', nn.ELU())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120)
        output = self.fc(output)
        return output

def featsLenet(outDim=3, pretrained=False):
    model = FeatsLeNet5(outDim=outDim)
    model_weights_random_xavier(model)
    if pretrained:
        modelFilename = '../../trainModels/models/modellenet5mnist_Iter6.torchmodel'
        initialize_pretrained_model(model, modelFilename)
    return model


def featsLenetOrig(pretrained=False):
    model = FeatsLeNet5Orig()
    model_weights_random_xavier(model)
    if pretrained:
        modelFilename = '../../trainModels/models/modellenet5mnist_Iter6.torchmodel'
        initialize_pretrained_model(model, modelFilename)
    return model


