import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from featuresModel import featsLenet
from helperFunctions import *
from losses import *


# parameters and names
case = "AugmentationNew"
outDim = 3
nAug = 0
delta = 5
trainstep = 2
transferTrainstep = 1
dataset = 'mnist'
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "../../data"

lamda = 1
featsPretrained = False
modelname = "DistLeNetNoNorm%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
model_folder = "trainedModels"

# model loading
featsModelname = "featsModel%s" % modelname
featsModel = load_model(featsLenet, model_folder, featsModelname, 0, featsPretrained, outDim)
freeze_layers(featsModel)
# remove last layer
nFeats = featsModel.fc[-1].in_features
nClasses = 10
featsModel.fc[-1] = torch.nn.Linear(nFeats, nClasses)
if transferTrainstep>=1:
    modelfilename = '%s/%sTransfer%s_Iter%i.state' % (model_folder, dataset, modelname, transferTrainstep)
    featsModel = load_model_weights(featsModel, modelfilename)

if torch.cuda.is_available():
    featsModel.cuda()

test_loader = load_mnist(datafolder, train_batch_size, train=False, download=False)
test_accuracy(featsModel, test_loader)