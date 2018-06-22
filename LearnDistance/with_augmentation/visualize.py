import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featuresModel

import sys
sys.path.insert(0, '../../trainModels')
#import lenet

trainstep = 4
delta = 50
lamda = 1 
batch_size = 1000
learningRate = 1e-3
#modelName = "LearnDistanceNoPretrainDistAlexNetAugmentationDelta%iLamda%iBatch%iLR%f" % (delta, lamda, batch_size, learningRate)
modelName = "LearnDistanceNoPretrainDistAlexNetAugmentationDelta%iLamda%iWeights01" % (delta, lamda)

modelfolder = "trainedModels"

modelfilename = '%s/featsModel%s_Iter%i' % (modelfolder, modelName, trainstep)
modelfile = torch.load(modelfilename+".state")
model = featuresModel()
model.load_state_dict(modelfile)
#modelfile = open(modelfilename+".torchmodel", 'rb')
#model = torch.load(modelfile, map_location=lambda storage, loc: storage)

Nsamples = 1000
Niter = 1



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

# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

writer = SummaryWriter(comment='%s_Iter%i_mnist_embedding' % (modelName, trainstep))

iterTrainLoader = iter(trainloader)

for i in range(Niter):

    input, label = next(iterTrainLoader)

    # forward
    output = model.forward(input)
    # output = model.convnet(input)
    output = torch.squeeze(output)
    print(output.size())
    output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
    #input = input.to(torch.device("cpu"))
    # save embedding
    writer.add_embedding(output, metadata=label.data, label_img=input.data, global_step=i)

writer.close()

