import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from featuresModel import featsLenetAE
from helperFunctions import *
from losses import *


# parameters and names
case = "Autoencoder"
outDim = 3
delta = 5
trainstep = 1
learningRate = 1e-3
dataset = 'mnist'
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train_batch_size = 1000
    Nsamples = int(60000 / train_batch_size)
    log_iter = int(Nsamples/2)
    Nepochs = 20
    datafolder = "/var/tmp/ioannis/data"
else:
    train_batch_size = 10
    Nsamples = int(60 / train_batch_size)
    log_iter = 10
    Nepochs = 1
    datafolder = "../../data"

lamda = 1
featsPretrained = False
distPretrained = False
modelname = "DistLeNet%sNoNormOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
log_name = "%sBatch%iLR%f_Iter%i" % (modelname, train_batch_size, learningRate, trainstep)
model_folder = "trainedModels"

train_loader = load_mnist(datafolder, train_batch_size, train=True, download=False)

# model loading
featsModelname = "featsModel%s" % modelname
featsModel = load_model(featsLenetAE, model_folder, featsModelname, trainstep-1, featsPretrained, outDim)

# optimizers
featsOptimizer = optim.Adam(featsModel.parameters(), lr=learningRate)

# writers and criterion
writer = SummaryWriter(comment='%s_loss_log' % (log_name))
criterion = torch.nn.MSELoss()

# Training
print('Start Training')
print(log_name)
global_step=0
for epoch in range(Nepochs):
    global_step += 1
    running_loss = 0.0
    iterTrainLoader = iter(train_loader)
    for i in range(Nsamples):
        input, _ = next(iterTrainLoader)
        inputAug = augment_batch(input)

        # transfer to cuda if available
        if torch.cuda.is_available():
            input = input.cuda()
            inputAug = inputAug.cuda()
            criterion.cuda()
        # print(inputAug)
        # zero the parameter gradients
        featsOptimizer.zero_grad()

        # optimize
        output = featsModel.forward(inputAug)
        loss = criterion(output, input)
        loss.backward()
        featsOptimizer.step()

        # print statistics
        running_loss += loss.item()
        writer.add_scalar(tag='MSELoss', scalar_value=loss.item(), global_step=global_step)
        if i % log_iter == log_iter-1:
            # print images to tensorboard
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')
print(log_name)

# save weights
save_model_weights(featsModel, model_folder, featsModelname, trainstep)
print('saved models')

writer.close()
