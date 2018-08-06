import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featsLenetFix, featsLenetFull, featsAE
from helperFunctions import *

import sys
sys.path.insert(0, '../../trainModels')

trainstep = 3
delta = 5
lamda = 1
Nsamples = 1000
nAug = 10

modelname = "featsModelLearnDistanceDistLeNetNoNormAugmentationDelta50Lamda1"
# modelname = "featsModelLearnDistanceDistLeNetNoNormAugmentation%iDelta%iLamda%i" % (nAug, delta, lamda)
modelfolder = "trainedModels"
featsModel = load_model(featsLenetFull, modelfolder, modelname, trainstep, pretrained=False)
featsModel.cpu()


transform = transforms.Compose(
    [transforms.ToTensor()])

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "../../data"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root=datafolder, train=True, download=False, transform=transform)
train_subset = torch.utils.data.dataset.Subset(train_dataset, range(Nsamples))
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=Nsamples, shuffle=False, num_workers=0)

test_dataset = torchvision.datasets.MNIST(root=datafolder, train=False, download=False, transform=transform)
test_subset = torch.utils.data.dataset.Subset(test_dataset, range(Nsamples))
testloader = torch.utils.data.DataLoader(test_subset, batch_size=Nsamples, shuffle=False, num_workers=0)


# Train Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_train' % (modelname, trainstep), log_dir='embeddings3')

iterTrainLoader = iter(trainloader)
input, label = next(iterTrainLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
# output = torch.cat((output, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data, global_step=14)

# Test Visualization
print('visualizing..')

iterTestLoader = iter(testloader)
input, label = next(iterTestLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
# output = torch.cat((output, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data, global_step=15)
writerEmb.close()
