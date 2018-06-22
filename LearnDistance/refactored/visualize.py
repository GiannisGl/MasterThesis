import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featuresModel
from helperFunctions import *

import sys
sys.path.insert(0, '../../trainModels')
import lenet

trainstep = 1
delta = 100
lamda = 1
Nsamples = 1000

name = "LearnDistanceNoPretrainDistAlexNetAugmentationDelta%iLamda%i_Iter%i" % (delta, lamda, trainstep)
modelfolder = "trainedModels"
modelfilename = '%s/featsModel%s' % (modelfolder, name)
modelfile = torch.load(modelfilename+".state")
model = featuresModel()
model.load_state_dict(modelfile)
model = load_model(featuresModelFull, trainstep, model_folder, modelname)




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

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


# Train Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_train' % (name, trainstep))

iterTrainLoader = iter(train_loader)
input, label = next(iterTrainLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()


# Test Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_test' % (log_name, trainstep))

iterTestLoader = iter(test_loader)
input, label = next(iterTestLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()
