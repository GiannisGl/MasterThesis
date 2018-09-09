import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch

from helperFunctions import augment_batch

data_folder = "../data"

# dataset = 'mnist'
dataset = 'cifar'
batchSize = 5

transform = transforms.Compose(
    [transforms.ToTensor()])

if dataset=='mnist':
    cmap = 'gray'
    train_set = torchvision.datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
elif dataset == 'cifar':
    cmap = None
    train_set = torchvision.datasets.CIFAR10(root=data_folder, train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize,
                                           shuffle=True, num_workers=0)

iterTrainLoader = iter(train_loader)
input, label = next(iterTrainLoader)
inputAug = augment_batch(input, dataset=dataset)
input = input.permute([0,2,3,1])
print(input.shape)
inputAug = inputAug.permute([0,2,3,1])
f, imgplot = plt.subplots(batchSize,2)
plt.axis("off")
for i in range(batchSize):
    imgplot[i,0].imshow(torch.squeeze(input[i]), cmap=cmap)
    imgplot[i,0].axis('off')
    imgplot[i,1].imshow(torch.squeeze(inputAug[i]), cmap=cmap)
    imgplot[i,1].axis('off')
plt.show()
