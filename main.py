from model import *
from loss import *
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


if torch.cuda.is_available():
    torch.cuda.set_device(0)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "./data"

trainset = torchvision.datasets.CIFAR10(root=datafolder, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

name = ""
trainstep = 1

if trainstep == 1:
    model = siamese_alexnet()
else:
    modelfilename = 'model%s_Iter%i.torchmodel' % (name,trainstep-1)
    modelfile = open(modelfilename, 'rb')
    model = torch.load(modelfile)

if torch.cuda.is_available():
    model.cuda()


optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

Nepochs = 3
Nsamples = 1
for epoch in range(Nepochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(Nsamples):
        iterTrainLoader = iter(trainloader)
        input1, _ = next(iterTrainLoader)
        input2, _ = next(iterTrainLoader)

        # wrap them in Variable
        input1 = Variable(input1)
        input2 = Variable(input2)

        if torch.cuda.is_available():
            input1.cuda()
            input2.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model.forward(input1,input2)
        output1 = outputs[0]
        output2 = outputs[1]
        if torch.cuda.is_available():
            output1 = output1.cuda()
            output2 = output2.cuda()

        loss = distance_loss(input1,input2,output1,output2)
        if torch.cuda.is_available():
            loss.cuda()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

modelfilename = 'model%s_Iter%i.torchmodel' % (name,trainstep)
modelfile = open(modelfilename, "wb")
torch.save(model, modelfile)
print('saved model')


# input, _ = next(iter(trainloader))
# input = Variable(input)
# output = model.forward_once(input)
# print(output.data)


# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# data = unpickle("data/cifar-10-batches-py/data_batch_1")
# print(list(data.keys()))