import torch.nn as nn
from helperFunctions import model_weights_random_xavier, initialize_pretrained_model
from torch import cat
from collections import OrderedDict


class DistanceAlexNet(nn.Module):

    def __init__(self):
        super(DistanceAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 1 * 1, 100),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(100, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 1),
            nn.ReLU(inplace=True)
        )


    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(x)
        return x

    def forward(self, input1, input2):
        input = cat((input1,input2),1)
        output = self.forward_once(input)
        return output


class DistanceLeNet5(nn.Module):

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
    # F7 - 1 (Output)

    def __init__(self):
        super(DistanceLeNet5, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(2, 6, kernel_size=(5, 5))),
            ('relu1', nn.ELU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ELU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ELU())
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ELU()),
            ('f7b', nn.Linear(84, 1))
            # no ELU on output
        ]))


    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 120 * 1 * 1)
        x = self.classifier(x)
        return x

    def forward(self, input1, input2):
        input = cat((input1,input2),1)
        output = self.forward_once(input)
        return output

def distanceModel(pretrained=False):
    model = DistanceLeNet5()
    model_weights_random_xavier(model)
    if pretrained:
        modelFilename = '../../trainModels/models/modellenet5mnist_Iter6.torchmodel'
        initialize_pretrained_model(model, modelFilename)
    return model
