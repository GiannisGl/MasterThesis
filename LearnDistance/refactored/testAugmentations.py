import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch

from helperFunctions import augment_batch

data_folder = "../data"


transform = transforms.Compose(
    [transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                       download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=True, num_workers=0)

iterTrainLoader = iter(train_loader)
input, label = next(iterTrainLoader)
batchSize = input.shape[0]
inputAug = augment_batch(input)
print(batchSize)
print(inputAug.shape)

f, imgplot = plt.subplots(3,2)
imgplot[0,0].imshow(torch.squeeze(input[0]))
imgplot[0,1].imshow(torch.squeeze(inputAug[0]))
imgplot[1,0].imshow(torch.squeeze(input[1]))
imgplot[1,1].imshow(torch.squeeze(inputAug[1]))
imgplot[2,0].imshow(torch.squeeze(input[2]))
imgplot[2,1].imshow(torch.squeeze(inputAug[2]))
plt.show()
