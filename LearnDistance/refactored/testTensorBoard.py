import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featsLenet, featsLenetFull, featsAE
from helperFunctions import *

import sys
sys.path.insert(0, '../../trainModels')

trainstep = 0
delta = 5
lamda = 1
Nsamples = 1000
nAug = 10

modelname = "featsModelLearnDistanceDistLeNetNoNormAugmentation%iDelta%iLamda%i" % (nAug, delta, lamda)
modelfolder = "trainedModels"
featsModel = load_model(featsLenetFix, modelfolder, modelname, trainstep, pretrained=True)
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

test_dataset = torchvision.datasets.MNIST(root=datafolder, train=True, download=False, transform=transform)
test_subset = torch.utils.data.dataset.Subset(test_dataset, range(Nsamples))
testloader = torch.utils.data.DataLoader(test_subset, batch_size=Nsamples, shuffle=False, num_workers=0)


# Train Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_train' % (modelname, trainstep), log_dir='embeddings10')

iterTrainLoader = iter(trainloader)
input, label = next(iterTrainLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
#output = torch.cat((output.item(), torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()


# Test Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_test' % (modelname, trainstep))

iterTestLoader = iter(testloader)
input, label = next(iterTestLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
#output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()
