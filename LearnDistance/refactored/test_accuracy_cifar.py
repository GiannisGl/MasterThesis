import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from featuresModel import featsLenet
from helperFunctions import *
from losses import *


# parameters and names
case = "CifarStrongerAug"
outDim = 3
nAug = 5
delta = 5
trainstep = 2
transferTrainstep = 1
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    batch_size = 1000
    datafolder = "/var/tmp/ioannis/data"
else:
    batch_size = 10
    datafolder = "../../data"

lamda = 1
featsPretrained = False
modelname = "DistInception%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
# modelname = "DistInception%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
# modelname = "DistInception%sOut%iDelta%i" % (case, outDim, delta)
model_folder = "trainedModels"

# model loading
featsModelname = "featsModel%s" % modelname
featsModel = load_model(featsLenet, model_folder, featsModelname, 0, featsPretrained, outDim)
freeze_layers(featsModel)
# remove last layer
nFeats = featsModel.fc[-1].in_features
nClasses = 10
featsModel.fc[-1] = torch.nn.Linear(nFeats, nClasses)
if transferTrainstep<1:
    modelfilename = '%s/%s_Iter%i.state' % (model_folder, featsModelname, trainstep)
    load_model_weights(featsModel, modelfilename)
else:
    modelfilename = '%s/%sTransferCifar_Iter%i.state' % (model_folder, modelname, transferTrainstep)
    featsModel = load_model_weights(featsModel, modelfilename)

if torch.cuda.is_available():
    featsModel.cuda()

test_loader = load_cifar(datafolder, batch_size, train=False, download=False)
test_accuracy(featsModel, test_loader)