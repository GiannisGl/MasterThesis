import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import sys
sys.path.insert(0, '../trainModels')
import lenet

# modelName = "LearnDistanceNorm01Delta10"
modelName = "LearnDistanceNoPretrain"
trainstep = 1
delta = 100

modelfolder = "trainedModels"

modelfilename = '%s/featsModel%sDelta%i_Iter%i.torchmodel' % (modelfolder, modelName, delta, trainstep)
# modelfilename = '../trainModels/models/modellenet5_Iter1.torchmodel'
modelfile = open(modelfilename, 'rb')
model = torch.load(modelfile, map_location=lambda storage, loc: storage)

batchSize = 100
Nsamples = 2



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "../data"

trainset = torchvision.datasets.MNIST(root=datafolder, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

writer = SummaryWriter(comment='mnist_embedding_delta%i__Iter%i' % (delta, trainstep))

iterTrainLoader = iter(trainloader)

for i in range(Nsamples):

    input, label = next(iterTrainLoader)

    # forward
    output = model.forward(input)
    # output = model.convnet(input)
    output = torch.squeeze(output)
    print(output.size())
    # output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
    input = input.to(torch.device("cpu"))
    # save embedding
    writer.add_embedding(output, metadata=label.data, label_img=input.data, global_step=i)

writer.close()

