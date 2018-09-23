import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from featuresModel import featsLenet
from distanceModel import distanceModel
from helperFunctions import *
from losses import *


# parameters and names
case = "SlackNew"
outDim = 3
nAug = 0
delta = 5
trainstep = 3
transferTrainstep = 1
dist = True
dataset = "mnist"
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    batch_size = 1000
    datafolder = "/var/tmp/ioannis/data"
else:
    batch_size = 10
    datafolder = "../../data"

lamda = 1
pretrained = False
modelname = "DistLeNetNoNorm%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
# modelname = "DistLeNet%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
# modelname = "DistLeNetNoNorm%sOut%iDelta%i" % (case, outDim, delta)
model_folder = "trainedModels"

# model loading
if dist:
    fullModelname = "distModel%s" % modelname
    model = load_model(distanceModel, model_folder, fullModelname, 0, pretrained)
else:
    fullModelname = modelname
    model = load_model(featsLenet, model_folder, fullModelname, 0, pretrained, outDim)
freeze_layers(model)
# remove last layer
nClasses = 10
if dist:
    nFeats = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(nFeats, nClasses)
else:
    nFeats = model.fc[-1].in_features
    model.fc[-1] = torch.nn.Linear(nFeats, nClasses)
if transferTrainstep<1:
    modelfilename = '%s/%s_Iter%i.state' % (model_folder, fullModelname, trainstep)
    load_model_weights(model, modelfilename)
else:
    modelfilename = '%s/%sTransfer%s_Iter%i_Iter%i.state' % (model_folder, dataset, fullModelname, trainstep, transferTrainstep)
    model = load_model_weights(model, modelfilename)

if torch.cuda.is_available():
    model.cuda()

test_loader = load_mnist(datafolder, batch_size, train=False, download=False)
print(modelfilename)
test_accuracy(model, test_loader, dist=dist)