from model import *
from loss import *
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

name = ""
trainstep = 1

modelfolder = "../models"

batchSize = 100
Nepochs = 3
Nsamples = 200

if torch.cuda.is_available():
    torch.cuda.set_device(1)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "./data"

trainset = torchvision.datasets.CIFAR10(root=datafolder, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


if trainstep == 1:
    model = siamese_alexnet()
else:
    modelfilename = 'model%s_Iter%i.torchmodel' % (name,trainstep-1)
    modelfile = open(modelfilename, 'rb')
    model = torch.load(modelfile)

if torch.cuda.is_available():
    model = model.cuda()


optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
criterion = distance_loss()

for epoch in range(Nepochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(Nsamples):
        iterTrainLoader = iter(trainloader)
        input1, _ = next(iterTrainLoader)
        input2, _ = next(iterTrainLoader)

        # wrap them in Variable
        if torch.cuda.is_available():
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
        else:
            input1 = Variable(input1,requires_grad=True)
            input2 = Variable(input2,requires_grad=True)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output1, output2 = model.forward(input1,input2)
        # output1 = outputs[0]
        # output2 = outputs[1]
        # wrap them in Variable
        if torch.cuda.is_available():
            output1 = output1.cuda()
            output2 = output2.cuda()
        else:
            output1 = output1
            output2 = output2

        # loss = distance_loss(input1,input2,output1,output2)
        # if torch.cuda.is_available():
            # criterion.cuda()

        loss = criterion(input1,input2,output1, output2)
        # print(loss.backward())
        loss.backward()
        optimizer.step()
        print(output2.data.size())

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

modelfilename = '%s/model%s_Iter%i.torchmodel' % (modelfolder,name,trainstep)
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
