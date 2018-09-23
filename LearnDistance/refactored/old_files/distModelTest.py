import torch
import torchvision
import torchvision.transforms as transforms
from distanceModel import distanceModel

import sys
sys.path.insert(0, '../../trainModels')
import lenet

trainstep = 1
delta = 50
lamda = 1
modelName = "LearnDistanceNoPretrainDistAlexNetAugmentationDelta%iLamda%i" % (delta, lamda)

modelfolder = "trainedModels"

modelfilename = '%s/distModel%s_Iter%i' % (modelfolder, modelName, trainstep)
if torch.cuda.is_available():
    modelfile = torch.load(modelfilename+".state")
else:
    modelfile = torch.load(modelfilename+".state", map_location=lambda storage, loc: storage)
model = distanceModel()
model.load_state_dict(modelfile)

batchSize = 100
Nsamples = 2

transform = transforms.Compose(
    [transforms.ToTensor()])

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "../../data"

trainset = torchvision.datasets.MNIST(root=datafolder, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

# writer = SummaryWriter(comment='%s_mnist_embedding' % (modelfilename))

iterTrainLoader = iter(trainloader)

for i in range(Nsamples):

    input1, label1 = next(iterTrainLoader)
    input2, label2 = next(iterTrainLoader)

    # forward
    output = model.forward(input1, input2)
    # output = model.convnet(input)
    output = torch.squeeze(output)
    print(output)
    # output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
    # input = input.to(torch.device("cpu"))
    # save embedding
    # writer.add_embedding(output, metadata=label.data, label_img=input.data, global_step=i)

# writer.close()

