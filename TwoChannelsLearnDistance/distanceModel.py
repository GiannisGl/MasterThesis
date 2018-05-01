import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch import cat


__all__ = ['siameseAlexNet', 'siamese_alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/trainedModels/alexnet-owt-4df8aa71.pth',
}


class DistanceAlexNet(nn.Module):

    def __init__(self):
        super(DistanceAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4 * 3 * 3, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 1),
        )


    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4 * 4 * 4)
        x = self.classifier(x)
        return x

    def forward(self, input1, input2):
        input = cat((input1,input2),0)
        output = self.forward_once(input)
        return output


def distanceModel(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DistanceAlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model