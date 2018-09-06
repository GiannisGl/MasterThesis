import torch
import torchvision
from tensorboardX import SummaryWriter
from inceptionModel import featsInception
from featuresModel import featsLenetOrig, featsLenet
from helperFunctions import *


trainstep = 2
case = "AugmentationNew"
# case = "Cifar"
outDim = 3
delta = 5
lamda = 1
Nsamples = 2000
dataset = 'mnist'
# dataset = 'cifar'

modelfolder = "trainedModels"
if dataset=='mnist':
    modelname = "featsModelDistLeNetNoNorm%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
    # modelname = "featsModelDistLeNet%sAug%iOut%iDelta%i" % (case, nAug, outDim, delta)
    featsModel = load_model(featsLenet, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)
elif dataset=='cifar':
    modelname = "featsModelDistInception%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
    #modelname = "featsModelDistInception%sAug%iOut%iDelta%i" % (case, nAug, outDim, delta)
    featsModel = load_model(featsInception, modelfolder, modelname, trainstep, pretrained=False, outDim=outDim)

#
if torch.cuda.is_available():
    # featsModel = featsModel.cuda()
    datafolder = "/var/tmp/ioannis/data"
else:
    # featsModel = featsModel.cpu()
    datafolder = "../../data"

#Train Visualization
print('visualizing..')
print('%s_Iter%i' %(modelname, trainstep))
writerEmb = SummaryWriter(comment='%s_Iter%i_embedding' % (modelname, trainstep))
visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=True)
visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=False)
writerEmb.close()




# writerEmb = SummaryWriter(comment='%sfirst2000subset' % dataset)
# visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=True, raw=True)
# visualize(writerEmb=writerEmb, model=featsModel, datafolder=datafolder, dataset=dataset, Nsamples=Nsamples, train=False, raw=True)
# writerEmb.close()