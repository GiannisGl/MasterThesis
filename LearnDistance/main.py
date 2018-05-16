import torch
import torch.optim as optim
import torchvision
#from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
from losses import *
from featuresModel import featuresModel
from distanceModel import distanceModel
from torch.autograd import Variable
from helperFunctions import *

name = "LearnDistanceNoPretrainDistLenet"
model_folder = "trainedModels"

trainstep = 1
batch_size = 1024
Nepochs = 2
Nsamples = 1000
learningRate = 1e-4
delta = 100
pretrained = False

if torch.cuda.is_available():
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../data"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1, 1, 1))])

train_set = torchvision.datasets.MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)

#train_val_length = len(train_val_set)
#train_percentage = 0.8
#train_length = floor(train_val_length*0.8)
#val_length = train_val_length-train_length
#train_set, val_set = random_split(train_val_set, [train_length, val_length])
# train_set = train_val_set


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0)

# testset = torchvision.datasets.MNIST(root=data_folder, train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


if trainstep == 1:
    featsModel = featuresModel(pretrained=pretrained)
    distModel = distanceModel(pretrained=pretrained)
    if not pretrained:
        model_weights_init(featsModel)
        model_weights_init(distModel)
else:
    featsModelfilename = '%s/featsModel%sDelta%i_Iter%i.torchmodel' % (model_folder, name, delta, trainstep - 1)
    distModelfilename = '%s/distModel%sDelta%i_Iter%i.torchmodel' % (model_folder, name, delta, trainstep - 1)
    featsModelfile = open(featsModelfilename, 'rb')
    distModelfile = open(distModelfilename, 'rb')
    featsModel = torch.load(featsModelfile)
    distModel = torch.load(distModelfile)

if torch.cuda.is_available():
    featsModel = featsModel.cuda()
    distModel = distModel.cuda()  
    torch.set_default_tensor_type('torch.cuda.FloatTensor')



featsOptimizer = optim.Adam(featsModel.parameters(), lr=learningRate, weight_decay=0.00001)
distOptimizer = optim.Adam(distModel.parameters(), lr=learningRate, weight_decay=0.00001)
criterion = distance_loss()
log_iter = 100
#writer = SummaryWriter(comment='LearnDistanceEmbedding')

# Train
for epoch in range(Nepochs):  # loop over the dataset multiple times

    torch.set_default_tensor_type('torch.FloatTensor')
    running_loss = 0.0
    for i in range(Nsamples):
        iterTrainLoader = iter(train_loader)
      
        input1, _ = next(iterTrainLoader)
        input2, _ = next(iterTrainLoader)
        input3, _ = next(iterTrainLoader)

        #label1 = Variable(label1, requires_grad=False)
        #label2 = Variable(label2, requires_grad=False)
        #label3 = Variable(label3, requires_grad=False)

        # wrap them in Variable
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            input3 = Variable(input3.cuda())
        else:
            input1 = Variable(input1,requires_grad=True)
            input2 = Variable(input2,requires_grad=True)
            input3 = Variable(input3,requires_grad=True)

        # zero the parameter gradients
        featsOptimizer.zero_grad()
        distOptimizer.zero_grad()

        if torch.cuda.is_available():
            criterion.cuda()

        loss = criterion(delta, input1, input2, input3, featsModel, distModel)
        loss.backward()
        featsOptimizer.step()
        distOptimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == 0:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')



featsModelfilename = '%s/featsModel%sDelta%i_Iter%i.torchmodel' % (model_folder, name, delta, trainstep)
distModelfilename = '%s/distModel%sDelta%i_Iter%i.torchmodel' % (model_folder, name, delta, trainstep)
featsModelfile = open(featsModelfilename, "wb")
distModelfile = open(distModelfilename, "wb")
torch.save(featsModel, featsModelfile)
torch.save(distModel, distModelfile)
print('saved models')
