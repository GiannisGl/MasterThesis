import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featsLenet, featsLenetFull, featsAE
from helperFunctions import *

import sys
sys.path.insert(0, '../../trainModels')
import lenet

trainstep = 5
delta = 50
lamda = 1
Nsamples = 1000

name = "LearnDistanceNoPretrainDistAlexNetAutoencoderDelta%iLamda%i_Iter%i" % (delta, lamda, trainstep)
modelfolder = "trainedModels"
modelfilename = '%s/featsModel%s' % (modelfolder, name)
modelfile = torch.load(modelfilename+".state")
featsModel = featsAE()
featsModel.load_state_dict(modelfile)
#model = load_model(model, modelname, model_folder, trainstep)




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "../../data"

trainset = torchvision.datasets.MNIST(root=datafolder, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Nsamples,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root=datafolder, train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=Nsamples,
                                         shuffle=True, num_workers=0)


# Train Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_train' % (name, trainstep))

iterTrainLoader = iter(trainloader)
input, label = next(iterTrainLoader)
output = featsModel.encoder(input)
output = torch.squeeze(output)
print(output.size())
#output = torch.cat((output.item(), torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()


# Test Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_test' % (name, trainstep))

iterTestLoader = iter(testloader)
input, label = next(iterTestLoader)
output = featsModel.encoder(input)
output = torch.squeeze(output)
print(output.size())
#output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()
