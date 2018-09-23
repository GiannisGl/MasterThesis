import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from featuresModel import featsLenet
from helperFunctions import *
from losses import *


# parameters and names
case = "MnistClustering"
outDim = 3
nAug = 3
delta = 5
trainstep = 4
learningRate = 1e-4
dataset = 'mnist'
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    train_batch_size = 1000
    Nsamples = int(60000 / (2*train_batch_size))
    log_iter = int(Nsamples/2)
    Nepochs = 50
    datafolder = "/var/tmp/ioannis/data"
else:
    train_batch_size = 10
    Nsamples = int(600 / (2*train_batch_size))
    log_iter = 10
    Nepochs = 1
    datafolder = "../../data"

featsPretrained = False
modelname = "DistLeNet%sAug%iOut%iDelta%i" % (case, nAug, outDim, delta)
log_name = "%sBatch%iLR%f_Iter%i" % (modelname, train_batch_size, learningRate, trainstep)
model_folder = "trainedModels"

train_loader = load_mnist(datafolder, train_batch_size, train=True, download=True)

# model loading
featsModelname = "featsModel%s" % modelname
featsModel = load_model(featsLenet, model_folder, featsModelname, trainstep-1, featsPretrained, outDim)

# optimizers
featsOptimizer = optim.Adam(featsModel.parameters(), lr=learningRate)

# writers and criterion
writer = SummaryWriter(comment='%s_loss_log' % (log_name))
criterion = distance_loss_ClusteringOnly(writer, log_iter, delta, nAug)

# Training
print('Start Training')
print(log_name)
for epoch in range(Nepochs):

    running_loss = 0.0
    iterTrainLoader = iter(train_loader)
    for i in range(Nsamples):
        input1, _ = next(iterTrainLoader)
        input2, _ = next(iterTrainLoader)

        # transfer to cuda if available
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            input1 = input1.cuda()
            input2 = input2.cuda()
            criterion.cuda()

        # zero the parameter gradients
        featsOptimizer.zero_grad()

        # optimize
        loss = criterion(input1, input2, featsModel)
        loss.backward()
        featsOptimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == log_iter-1:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')
print(log_name)

# save weights
save_model_weights(featsModel, model_folder, featsModelname, trainstep)
print('saved models')

writer.close()