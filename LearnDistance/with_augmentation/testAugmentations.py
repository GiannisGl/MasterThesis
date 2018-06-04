import torch
import torchvision
import torchvision.transforms as transforms
from helperFunctions import *
import matplotlib.pyplot as plt

data_folder = "../data"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

train_set = torchvision.datasets.MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=True, num_workers=0)


iterTrainLoader = iter(train_loader)
input1, _ = next(iterTrainLoader)
input1Aug = random_augmentation(input1)

f, imgplot = plt.subplots(1,2)
imgplot[0].imshow(torch.squeeze(input1), cmap='gray')
imgplot[1].imshow(torch.squeeze(input1Aug), cmap='gray')
plt.show()
