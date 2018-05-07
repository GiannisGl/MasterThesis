import torch
from math import floor
import torch.optim as optim
import torchvision
#from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
#from TwoChannelsLearnDistance.losses import *
from losses import *
#from TwoChannelsLearnDistance.featuresModel import featuresModel
from featuresModel import featuresModel
#from TwoChannelsLearnDistance.distanceModel import distanceModel
from distanceModel import distanceModel
#from tensorboardX import SummaryWriter
from torch.autograd import Variable

name = "TwoChannelsLearnDistance"
model_folder = "trainedModels"

trainstep = 1
batch_size = 100
Nepochs = 5
Nsamples = 1000


if torch.cuda.is_available():
    torch.cuda.set_device(1)
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../data"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((-1, -1, -1), (1, 1, 1))])

train_val_set = torchvision.datasets.MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)
#train_val_length = len(train_val_set)
#train_percentage = 0.8
#train_length = floor(train_val_length*0.8)
#val_length = train_val_length-train_length
#train_set, val_set = random_split(train_val_set, [train_length, val_length])
train_set = train_val_set


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0)

# testset = torchvision.datasets.MNIST(root=data_folder, train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


if trainstep == 1:
    featsModel = featuresModel()
    distModel = distanceModel()
else:
    modelfilename = '%s/model%s_Iter%i.torchmodel' % (model_folder, name, trainstep - 1)
    modelfile = open(modelfilename, 'rb')
    model = torch.load(modelfile)

if torch.cuda.is_available():
    featsModel = featsModel.cuda()
    distModel = distModel.cuda()  
    torch.set_default_tensor_type('torch.cuda.FloatTensor')



featsOptimizer = optim.Adam(featsModel.parameters(), lr=0.001, weight_decay=0.00001)
distOptimizer = optim.Adam(distModel.parameters(), lr=0.001, weight_decay=0.00001)
criterion = distance_loss()
log_iter = 100
#writer = SummaryWriter(comment='LearnDistanceEmbedding')

# Train
for epoch in range(Nepochs):  # loop over the dataset multiple times

    torch.set_default_tensor_type('torch.FloatTensor')
    iterTrainLoader = iter(train_loader)
    running_loss = 0.0
    for i in range(Nsamples):
        #n_iter = (epoch * len(train_loader)) + i
      
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

        # forward + backward + optimize
        # output1, output2 = model.forward(input1,input2)
        # output1 = outputs[0]
        # output2 = outputs[1]
        # wrap them in Variable
        # if torch.cuda.is_available():
        #     output1 = output1.cuda()
        #     output2 = output2.cuda()

        # loss = distance_loss(input1,input2,output1,output2)
        if torch.cuda.is_available():
            criterion.cuda()

        loss = criterion(input1, input2, input3, featsModel, distModel)
        loss.backward()
        featsOptimizer.step()
        distOptimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == 0:    # print every embedding_log mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
#            writer.add_embedding(output1.data, metadata=label1.data, label_img=input1.data, global_step=2*n_iter)
#            writer.add_embedding(output2.data, metadata=label2.data, label_img=input2.data, global_step=2*n_iter+1)


print('Finished Training')


#writer.close()


modelfilename = '%s/model%s_Iter%i.torchmodel' % (model_folder, name, trainstep)
modelfile = open(modelfilename, "wb")
torch.save(model, modelfile)
print('saved model')
