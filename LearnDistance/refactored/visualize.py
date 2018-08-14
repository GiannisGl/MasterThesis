import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from featuresModel import featsLenetOrig, featsLenet
from helperFunctions import *


trainstep = 3
case = "Augmentation"
outDim = 2
delta = 5
lamda = 1
Nsamples = 2000
dataset = 'mnist'
# dataset = 'cifar'

# modelname = "featsModelDistLeNetNoNormSlackOut3Delta10Lamda1"
modelname = "featsModelDistLeNetNoNorm%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
modelfolder = "trainedModels"
# modelfilename = '%s/featsModel%s' % (modelfolder, name)
# modelfile = torch.load(modelfilename+".state")
# featsModel = featsLenetFull()
# featsModel.load_state_dict(modelfile)
featsModel = load_model(featsLenet, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)
featsModel.cpu()

if torch.cuda.is_available():
    datafolder = "/var/tmp/ioannis/data"
else:
    datafolder = "../../data"

# Train Visualization
print('visualizing..')
print('%s_Iter%i' %(modelname, trainstep))
writerEmb = SummaryWriter(comment='%s_Iter%i_embedding' % (modelname, trainstep))
visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=True)
visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=False)
writerEmb.close()
