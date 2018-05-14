from collections import OrderedDict

import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../trainModels')
import lenet


# __all__ = ['FeatsAlexNet']
#
#
# model_urls = {
#     'alexnet': 'https://download.pytorch.org/trainedModels/alexnet-owt-4df8aa71.pth',
# }


class FeatsAlexNet(nn.Module):

    def __init__(self):
        super(FeatsAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
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
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(x)
        return x

    # def forward(self, input1, input2):
    #     input = cat((input1,input2),0)
    #     output = self.forward_once(input)
    #     return output


class FeatsLeNet5(nn.Module):
    """
    Input - 1x28x28
    C1 - 6@24x24 (5x5 kernel)
    tanh
    S2 - 6@12x12 (2x2 kernel, stride 2) Subsampling
    C3 - 16@8x8 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@4x4 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(FeatsLeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6b', nn.Linear(120, 3)),
            ('sig6b', nn.LogSoftmax(0))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120)
        output = self.fc(output)
        return output

def featuresModel(pretrained=False, **kwargs):
    model = FeatsLeNet5(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        modelFilename = '../trainModels/models/modellenet5_Iter4.torchmodel'
        pretrained = torch.load(modelFilename)
        pretrained_dict = pretrained.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
