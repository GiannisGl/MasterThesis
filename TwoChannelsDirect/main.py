import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from loss import *
from model import *
from pretrainedModel import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable

name = "2channelsTensorboardMNIST"
trainstep = 1

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

trainset = torchvision.datasets.MNIST(root=datafolder, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=False, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


pretrainedModel = mnist(pretrained=True)

if trainstep == 1:
    model = siamese_alexnet()
else:
    modelfilename = '%s/model%s_Iter%i.torchmodel'     % (modelfolder,name,trainstep-1)
    modelfile = open(modelfilename, 'rb')
    model = torch.load(modelfile)

if torch.cuda.is_available():
    model = model.cuda()
    pretrainedModel = pretrainedModel.cuda()


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
criterion = distance_loss()


embedding_log = 5
writer = SummaryWriter(comment='cifar10_embedding_training')

# Train
for epoch in range(Nepochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(Nsamples):
        n_iter = (epoch*len(trainloader))+i

        iterTrainLoader = iter(trainloader)
        input1, label1 = next(iterTrainLoader)
        input2, label2 = next(iterTrainLoader)

        # wrap them in Variable
        if torch.cuda.is_available():
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
        else:
            input1 = Variable(input1,requires_grad=True)
            input2 = Variable(input2,requires_grad=True)

        label1 = Variable(label1, requires_grad=False)
        label2 = Variable(label2, requires_grad=False)

        # get features for input distance
        input1feats = pretrainedModel.features(input1)
        input2feats = pretrainedModel.features(input2)

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

        # loss = distance_loss(input1,input2,output1,output2)
        # if torch.cuda.is_available():
            # criterion.cuda()

        loss = criterion(input1feats,input2feats,output1, output2)
        loss.backward()
        optimizer.step()

        #LOGGING
        writer.add_scalar('loss', loss.data[0], n_iter)

        # print statistics
        running_loss += loss.data[0]
        if i % embedding_log == 0:    # print every embedding_log mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            writer.add_embedding(output1.data, metadata=label1.data, label_img=input1.data, global_step=2*n_iter)
            writer.add_embedding(output2.data, metadata=label2.data, label_img=input2.data, global_step=2*n_iter+1)

print('Finished Training')

writer.close()


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
