import torch
import torchvision
from tensorboardX import SummaryWriter
from inceptionModel import featsInception, featsInceptionAE
from helperFunctions import *


trainstep = 2
case = "CifarExactWithClustering"
outDim = 3
nAug = 0
delta = 5
lamda = 1
Nsamples = 2000
dataset = 'cifar'

modelfolder = "trainedModels"

if case=="Autoencoder":
    ae = True
    modelname = "featsModelDistInception%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
    featsModel = load_model(featsInceptionAE, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)
else:
    ae = False
    modelname = "featsModelDistInception%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
    featsModel = load_model(featsInception, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)


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
