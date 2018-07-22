import torch
import torchvision
import torchvision.transforms as transforms
from augmentation import *
import matplotlib.pyplot as plt

data_folder = "../data"


transform = transforms.Compose(
    [transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000,
                                           shuffle=True, num_workers=0)


transformAug = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomAffine(scale=[0.8, 1.1], degrees=10, translate=[0.2, 0.2], shear=10), transforms.ToTensor()])

iterTrainLoader = iter(train_loader)
input, label = next(iterTrainLoader)
batchSize = input.shape[0]
inputAug = torch.Tensor(input.shape)
print(batchSize)
print(inputAug.shape)
for i in range(batchSize):
    inputAug[i] = transformAug(input[i].clone())


f, imgplot = plt.subplots(3,2)
imgplot[0,0].imshow(torch.squeeze(input[0]), cmap='gray')
imgplot[0,1].imshow(torch.squeeze(inputAug[0]), cmap='gray')
imgplot[1,0].imshow(torch.squeeze(input[1]), cmap='gray')
imgplot[1,1].imshow(torch.squeeze(inputAug[1]), cmap='gray')
imgplot[2,0].imshow(torch.squeeze(input[2]), cmap='gray')
imgplot[2,1].imshow(torch.squeeze(inputAug[2]), cmap='gray')
plt.show()
