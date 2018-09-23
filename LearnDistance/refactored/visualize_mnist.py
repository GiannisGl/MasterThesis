import torch
import torchvision
from tensorboardX import SummaryWriter
from featuresModel import featsLenetOrig, featsLenet, featsLenetAE
from helperFunctions import *


trainstep = 2
case = "MnistExactWithClustering"
outDim = 3
nAug = 0
delta = 5
lamda = 1
Nsamples = 2000
dataset = 'mnist'

modelfolder = "trainedModels"

if case=="Autoencoder":
    ae = True
    modelname = "featsModelDistLeNet%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
    featsModel = load_model(featsLenetAE, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)
else:
    ae = False
    modelname = "featsModelDistLeNet%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
    featsModel = load_model(featsLenet, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)


if torch.cuda.is_available():
    featsModel = featsModel.cuda()
    datafolder = "/var/tmp/ioannis/data"
else:
    featsModel = featsModel.cpu()
    datafolder = "../../data"

#Train Visualization
print('visualizing..')
print('%s_Iter%i' %(modelname, trainstep))
writerEmb = SummaryWriter(comment='%s_Iter%i_embedding' % (modelname, trainstep))
visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=True, ae=ae)
visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=False, ae=ae)
writerEmb.close()