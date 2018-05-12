import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable

name = "2channels"
trainstep = 6

modelfolder = "models"

batchSize = 50
Nepochs = 5
Nsamples = 100

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "../data"

trainset = torchvision.datasets.CIFAR10(root=datafolder, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


if trainstep == 1:
    model = siamese_alexnet()
else:
    modelfilename = '%s/model%s_Iter%i.torchmodel'     % (modelfolder,name,trainstep-1)
    modelfile = open(modelfilename, 'rb')
    model = torch.load(modelfile)

if torch.cuda.is_available():
    model = model.cuda()

writer = SummaryWriter(comment='cifar10_embedding_training')

iterTrainLoader = iter(trainloader)

for i in range(Nsamples):

    input, label = next(iterTrainLoader)

    # w;dföösdfsdfggrap them in Variable
    if torch.cuda.is_available():
        input = Variable(input.cuda())
    else:
        input = Variable(input,requires_grad=True)

    label = Variable(label, requires_grad=False)

    # forward + backward + optimize
    output = model.forward_once(input)
    # wrap them in Variable
    #if torch.cuda.is_available():
    #    output = output.cuda()

    output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
    input = input.to(torch.device("cpu"))
    # save embedding
    writer.add_embedding(output, metadata=label.data, label_img=input.data, global_step=i)

writer.close()
