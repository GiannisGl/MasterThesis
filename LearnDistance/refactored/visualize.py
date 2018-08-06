import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featsLenetFix, featsLenetFull, featsAE
from helperFunctions import *


trainstep = 24
delta = 50
lamda = 1
Nsamples = 2000
nAug = 10

modelname = "featsModelLearnDistanceDistLeNetNoNormpartFixFeatsDelta5Lamda1distFix"
# modelname = "featsModelLearnDistanceDistLeNetNoNormAugmentation%iDelta%iLamda%i" % (nAug, delta, lamda)
modelfolder = "trainedModels"
# modelfilename = '%s/featsModel%s' % (modelfolder, name)
# modelfile = torch.load(modelfilename+".state")
# featsModel = featsLenetFull()
# featsModel.load_state_dict(modelfile)
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
train_subset = torch.utils.data.dataset.Subset(train_dataset, range(Nsamples, 2*Nsamples))
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=Nsamples, shuffle=False, num_workers=0)

test_dataset = torchvision.datasets.MNIST(root=datafolder, train=False, download=False, transform=transform)
test_subset = torch.utils.data.dataset.Subset(test_dataset, range(Nsamples, 2*Nsamples))
testloader = torch.utils.data.DataLoader(test_subset, batch_size=Nsamples, shuffle=False, num_workers=0)


# Train Visualization
global_step = 3
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_train' % (modelname, trainstep))

iterTrainLoader = iter(trainloader)
input, label = next(iterTrainLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
#output = torch.cat((output.item(), torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, label_img=input, global_step=2*global_step)
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
writerEmb.add_embedding(output, label_img=input, global_step=2*global_step+1)
writerEmb.close()
