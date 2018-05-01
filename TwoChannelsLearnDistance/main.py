# import torch
from math import floor
import torch.optim as optim
import torchvision
import torch.utils.data.dataset
import torchvision.transforms as transforms
from TwoChannelsLearnDistance.losses import *
from TwoChannelsLearnDistance.models import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable

name = "TwoChannelsLearnDistance"
model_folder = "models"

trainstep = 1
batch_size = 100
Nepochs = 5
Nsamples = 100


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../data"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((-1, -1, -1), (1, 1, 1))])

train_val_set = torchvision.datasets.MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)
train_val_length = len(train_val_set)
train_percentage = 0.8
train_length = floor(train_val_length*0.8)
val_length = train_val_length-train_length
train_set, val_set = random_split(train_val_set, [train_length, val_length])


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0)

# testset = torchvision.datasets.MNIST(root=data_folder, train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


if trainstep == 1:
    model = siamese_alexnet()
else:
    modelfilename = '%s/model%s_Iter%i.torchmodel'     % (model_folder, name, trainstep - 1)
    modelfile = open(modelfilename, 'rb')
    model = torch.load(modelfile)

if torch.cuda.is_available():
    model = model.cuda()
    pretrainedModel = pretrainedModel.cuda()


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
criterion = distance_loss()
embedding_log = 5
writer = SummaryWriter(comment='LearnDistanceEmbedding')

# Train
for epoch in range(Nepochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(Nsamples):
        n_iter = (epoch * len(train_loader)) + i

        iterTrainLoader = iter(train_loader)
        input1, label1 = next(iterTrainLoader)
        input2, label2 = next(iterTrainLoader)
        input3, label3 = next(iterTrainLoader)

        label1 = Variable(label1, requires_grad=False)
        label2 = Variable(label2, requires_grad=False)
        label3 = Variable(label3, requires_grad=False)

        # wrap them in Variable
        if torch.cuda.is_available():
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            input2 = Variable(input3.cuda())
        else:
            input1 = Variable(input1,requires_grad=True)
            input2 = Variable(input2,requires_grad=True)
            input2 = Variable(input3,requires_grad=True)

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

print('Finished Training')


writer.close()


modelfilename = '%s/model%s_Iter%i.torchmodel' % (model_folder, name, trainstep)
modelfile = open(modelfilename, "wb")
torch.save(model, modelfile)
print('saved model')
