import datetime
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from augmentation import *
from featuresModel import featsLenet, featsLenetFull
from distanceModel import distanceModel
from helperFunctions import *
from losses import *


# parameters and names
nAug = 10
case = "Augmentation%i" % nAug
trainstep = 1
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    train_batch_size = 1000
    Nepochs = 50
else:
    train_batch_size = 10
    Nepochs = 1
Nsamples = int(600 / (3*train_batch_size))
learningRate = 1e-3
delta = 5
lamda = 1
log_iter = int(Nsamples/2)
featsPretrained = False
distPretrained = False
modelname = "LearnDistanceDistLeNetNoNorm%sDelta%iLamda%i" % (case, delta, lamda)
log_name = "%sBatch%iLR%f_Iter%i" % (modelname, train_batch_size, learningRate, trainstep)
model_folder = "trainedModels"

# dataset loading
if torch.cuda.is_available():
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../../data"
train_loader = load_mnist(data_folder, train_batch_size, train=True, download=False)

# model loading
featsModelname = "featsModel%s" % modelname
featsModel = load_model(featsLenetFull, model_folder, featsModelname, trainstep-1, featsPretrained)
distModelname = "distModel%s" % modelname
distModel = load_model(distanceModel, model_folder, distModelname, trainstep-1, distPretrained)

# optimizers
featsOptimizer = optim.Adam(featsModel.parameters(), lr=learningRate)
distOptimizer = optim.Adam(distModel.parameters(), lr=learningRate)

# writers and criterion
writer = SummaryWriter(comment='%s_loss_log' % (log_name))
writer_img = SummaryWriter(comment='%s_images' % (log_name))
criterion = distance_loss(writer, writer_img, log_iter, delta, lamda, nAug)

# Training
print('Start Training')
print(log_name)
for epoch in range(Nepochs):

    running_loss = 0.0
    iterTrainLoader = iter(train_loader)
    for i in range(Nsamples):
        input1, _ = next(iterTrainLoader)
        input2, _ = next(iterTrainLoader)
        input3, _ = next(iterTrainLoader)

        # transfer to cuda if available
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            input1 = input1.cuda()
            input2 = input2.cuda()
            input3 = input3.cuda()
            criterion.cuda()

        # zero the parameter gradients
        featsOptimizer.zero_grad()
        distOptimizer.zero_grad()

        # optimize
        loss = criterion(input1, input2, input3, featsModel, distModel)
        loss.backward()
        distOptimizer.step()
        featsOptimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == log_iter-1:
            # print images to tensorboard
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')

# save weights
save_model_weights(featsModel, model_folder, featsModelname, trainstep)
save_model_weights(distModel, model_folder, distModelname, trainstep)
print('saved models')


