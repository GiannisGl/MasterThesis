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
from augmentation import *
from tensorboardX import SummaryWriter
import datetime

case = "Augmentation"
trainstep = 1
# Per Epoch one iteration over the dataset
train_batch_size = 100
test_batch_size = 1000
Nsamples = int(60000 / train_batch_size)
Nepochs = 1
learningRate = 1e-3
delta = 100
lamda = 1
log_iter = 100
featsPretrained = False
distPretrained = False

curDatetime = datetime.datetime.now().isoformat();

modelname = "%sLearnDistanceNoPretrainDistAlexNet%sDelta%iLamda%i" % (curDatetime, case, delta, lamda)
log_name = "%sBatch%iLR%f_Iter%i" % (modelname, train_batch_size, learningRate, trainstep)
model_folder = "trainedModels"

if torch.cuda.is_available():
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../../data"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                           shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root=data_folder, train=False,
                                      download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                          shuffle=False, num_workers=0)

featsModelname = "featsModel%s" % modelname
featsModel = load_model(featuresModel, featsPretrained, trainstep-1, model_folder, featsModelname)
distModelname = "distModel%s" % modelname
distModel = load_model(distanceModel, distPretrained, trainstep-1, model_folder, distModelname)

#
# featsModel = featuresModel(pretrained=featsPretrained)
# distModel = distanceModel(pretrained=distPretrained)
# if trainstep <= 1:
#     if not featsPretrained:
#         model_weights_init(featsModel)
#     if not distPretrained:
#         model_weights_init(distModel)
# else:
#     # load featsModel
#     load_model(feat)
#     featsModelfilename = '%s/featsModel%s_Iter%i.state' % (model_folder, modelname, trainstep - 1)
#     featsModelfile = torch.load(featsModelfilename)
#     featsModel.load_state_dict(featsModelfile)
#     # load distModel
#     distModelfilename = '%s/distModel%s_Iter%i.state' % (model_folder, modelname, trainstep - 1)
#     distModelfile = torch.load(distModelfilename)
#     distModel.load_state_dict(distModelfile)

if torch.cuda.is_available():
    featsModel = featsModel.cuda()
    distModel = distModel.cuda()  
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


featsOptimizer = optim.Adam(featsModel.parameters(), lr=learningRate, weight_decay=0.00001)
distOptimizer = optim.Adam(distModel.parameters(), lr=learningRate, weight_decay=0.00001)
writer = SummaryWriter(comment='%s_Iter%i_loss_log' % (log_name, trainstep))
criterion = distance_loss(writer, delta, lamda)


print('Start Training')
print("%s_Iter%i" % (log_name, trainstep))
# Train
for epoch in range(Nepochs):  # loop over the dataset multiple times

    torch.set_default_tensor_type('torch.FloatTensor')
    running_loss = 0.0
    iterTrainLoader = iter(train_loader)
    for i in range(Nsamples):
        input1, _ = next(iterTrainLoader)
        input2, _ = next(iterTrainLoader)
        input3, _ = next(iterTrainLoader)

        #label1 = Variable(label1, requires_grad=False)
        #label2 = Variable(label2, requires_grad=False)
        #label3 = Variable(label3, requires_grad=False)

        # wrap them in Variable
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            input1 = input1.cuda()
            input2 = input2.cuda()
            input3 = input3.cuda()

        # zero the parameter gradients
        featsOptimizer.zero_grad()
        distOptimizer.zero_grad()

        if torch.cuda.is_available():
            criterion.cuda()

        loss = criterion(input1, input2, input3, featsModel, distModel)
        loss.backward()
        distOptimizer.step()
        featsOptimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == log_iter-1:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')


writer.close()


save_model_weights(featsModel, model_folder, featsModelname, trainstep)
save_model_weights(distModel, model_folder, distModelname, trainstep)
# featsModelfilename = '%s/featsModel%s_Iter%i.state' % (model_folder, modelname, trainstep)
# distModelfilename = '%s/distModel%s_Iter%i.state' % (model_folder, modelname, trainstep)
# featsModelfile = open(featsModelfilename, "wb")
# distModelfile = open(distModelfilename, "wb")
# torch.save(featsModel.state_dict(), featsModelfile)
# torch.save(distModel.state_dict(), distModelfile)
print('saved models')


# Train Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_train' % (log_name, trainstep))

iterTrainLoader = iter(train_loader)
input, label = next(iterTrainLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()


# Test Visualization
print('visualizing..')
writerEmb = SummaryWriter(comment='%s_Iter%i_mnist_embedding_test' % (log_name, trainstep))

iterTestLoader = iter(test_loader)
input, label = next(iterTestLoader)
output = featsModel.forward(input)
output = torch.squeeze(output)
print(output.size())
output = torch.cat((output.data, torch.ones(len(output), 1)), 1)
# input = input.to(torch.device("cpu"))
# save embedding
writerEmb.add_embedding(output, metadata=label.data, label_img=input.data)
writerEmb.close()


