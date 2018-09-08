import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from inceptionModel import featsInception, distInception
from helperFunctions import *
from losses import *


# parameters and names
case = "CifarStrongerAug"
outDim = 3
nAug = 5
delta = 5
trainstep = 2
transferTrainstep = 1
dist = False
dataset = "cifar"
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
if dist:
    fullModelname = "distModel%s" % modelname
    model = load_model(distInception, model_folder, fullModelname, 0, pretrained)
else:
    fullModelname = modelname
    model = load_model(featsInception, model_folder, fullModelname, 0, pretrained, outDim)
freeze_layers(model)
# remove last layer
nClasses = 10
nFeats = model.classifier.in_features
model.classifier = torch.nn.Linear(nFeats, nClasses)
if transferTrainstep<1:
    modelfilename = '%s/%s_Iter%i.state' % (model_folder, fullModelname, trainstep)
    load_model_weights(model, modelfilename)
else:
    modelfilename = '%s/%sTransfer%s_Iter%i_Iter%i.state' % (model_folder, dataset, fullModelname, trainstep, transferTrainstep)
    model = load_model_weights(model, modelfilename)

if torch.cuda.is_available():
    featsModel.cuda()

test_loader = load_cifar(datafolder, batch_size, train=False, download=False, shuffle=False)
test_accuracy(featsModel, test_loader)