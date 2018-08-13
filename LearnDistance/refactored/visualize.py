import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featsLenetOrig, featsLenet
from helperFunctions import *


trainstep = 1
case = "Augmentation"
outDim = 3
delta = 5
lamda = 1
Nsamples = 2000

# modelname = "featsModelDistLeNetNoNormSlackOut3Delta10Lamda1"
modelname = "featsModelDistLeNetNoNorm%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
modelfolder = "trainedModels"
# modelfilename = '%s/featsModel%s' % (modelfolder, name)
# modelfile = torch.load(modelfilename+".state")
# featsModel = featsLenetFull()
# featsModel.load_state_dict(modelfile)
featsModel = load_model(featsLenet, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)
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
print('%s_Iter%i' %(modelname, trainstep))
writerEmb = SummaryWriter(comment='%s_Iter%i_embedding' % (modelname, trainstep))

iterTrainLoader = iter(trainloader)
input, label = next(iterTrainLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print('train: %s' % list(output.size()))
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, label_img=input, metadata=label, tag="1.train")

# Test Visualization
iterTestLoader = iter(testloader)
input, label = next(iterTestLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print('test: %s' % list(output.size()))
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, label_img=input, metadata=label, tag="2.test")
writerEmb.close()
