import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable

# modelName = "LearnDistanceNorm01Delta10"
modelName = "LearnDistance"
trainstep = 1

modelfolder = "trainedModels"

modelfilename = '%s/featsModel%s_Iter%i.torchmodel' % (modelfolder, modelName, trainstep)
modelfile = open(modelfilename, 'rb')
model = torch.load(modelfile, map_location=lambda storage, loc: storage)

batchSize = 100
Nsamples = 1



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((-1, -1, -1), (1, 1, 1))])

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

writer = SummaryWriter(comment='mnist_embedding_training')

iterTrainLoader = iter(trainloader)

for i in range(Nsamples):

    input, label = next(iterTrainLoader)

    # forward
    output = model.forward(input)

    output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
    input = input.to(torch.device("cpu"))
    # save embedding
    writer.add_embedding(output, metadata=label.data, label_img=input.data, global_step=i)

writer.close()
