import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from inceptionModel import featsInception
from helperFunctions import *
from losses import *


# parameters and names
case = "CifarNewNoAug"
#case = "CifarClusteringNew"
outDim = 3
nAug = 0
delta = 5
trainstep = 2
transferTrainstep = 0
learningRate = 1e-3
dataset = 'cifar'
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    train_batch_size = 1000
    Nsamples = int(50000 / train_batch_size)
    log_iter = int(Nsamples/2)
    Nepochs = 50
    datafolder = "/var/tmp/ioannis/data"
else:
    train_batch_size = 100
    Nsamples = int(600 / train_batch_size)
    log_iter = 1
    Nepochs = 1
    datafolder = "../../data"

lamda = 1
featsPretrained = False
modelname = "DistInception%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
#modelname = "DistInception%sAug%iOut%iDelta%i" % (case, nAug, outDim, delta)
log_name = "featsTransfer%s%sAug%iBatch%iLR%f_Iter%i_Iter%i" % (dataset, modelname, nAug, train_batch_size, learningRate, trainstep, transferTrainstep)
model_folder = "trainedModels"

train_loader = load_cifar(datafolder, train_batch_size, train=True, download=False)

# model loading
featsModelname = "featsModel%s" % modelname
featsModel = load_model(featsInception, model_folder, featsModelname, 0, featsPretrained, outDim)
if transferTrainstep<1:
    featsModel = load_model(featsInception, model_folder, featsModelname, trainstep, featsPretrained, outDim)
freeze_layers(featsModel)
# remove last layer
nFeats = featsModel.fc.in_features
nClasses = 10
featsModel.fc = torch.nn.Linear(nFeats, nClasses)
if transferTrainstep>=1:
    modelfilename = '%s/%sTransfer%s_Iter%i_Iter%i.state' % (model_folder, dataset, modelname, trainstep, transferTrainstep)
    featsModel = load_model_weights(featsModel, modelfilename)
if torch.cuda.is_available():
    featsModel.cuda()

# optimizers
featsOptimizer = optim.Adam(featsModel.fc.parameters(), lr=learningRate)
criterion = torch.nn.CrossEntropyLoss()

# writers and criterion
writer = SummaryWriter(comment='%s_loss_log' % (log_name))

# Training
print('Start Training')
print(log_name)
for epoch in range(Nepochs):

    running_loss = 0.0
    iterTrainLoader = iter(train_loader)
    for i in range(Nsamples):
        input, label = next(iterTrainLoader)

        # transfer to cuda if available
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            input = input.cuda()
            label = label.cuda()
            criterion.cuda()

        # zero the parameter gradients
        featsOptimizer.zero_grad()

        # optimize
        output = featsModel.forward(input)
        loss = criterion(output, label)
        loss.backward()
        featsOptimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == log_iter-1:
            # print images to tensorboard
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')
print(log_name)

# save weights
transferModelname = "%sTransfer%s_Iter%i" % (dataset, modelname, trainstep)
print(transferModelname)
save_model_weights(featsModel, model_folder, transferModelname, transferTrainstep+1)
print('saved models')

writer.close()

test_loader = load_cifar(datafolder, train_batch_size, train=False, download=False)
test_accuracy(featsModel, test_loader)