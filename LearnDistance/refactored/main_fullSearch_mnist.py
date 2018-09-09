import torch
from tensorboardX import SummaryWriter
from distanceModel import distanceModel
from featuresModel import featsLenet, featsLenetAE
from helperFunctions import *
from losses import *


# parameters and names
case = "Autoencoder"
outDim = 3
nAug = 3
delta = 5
trainstep = 4
dataset = 'mnist'
# Per Epoch one iteration over the dataset
N_test_samples = 10000
dataset_size = 60000
dist = False
ae = True
if torch.cuda.is_available():
    train_batch_size = 60000
    N_train_batches = int(dataset_size/train_batch_size)
    datafolder = "/var/tmp/ioannis/data"
else:
    train_batch_size = 100
    N_train_batches = 20
    N_test_samples = 5
    datafolder = "../../data"

lamda = 1
pretrained = False
#modelname = "DistLeNetNoNorm%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
modelname = "DistLeNetNoNorm%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
log_name = "fullSearch%s%s_Iter%i" % (dataset, modelname, trainstep)
model_folder = "trainedModels"

search_train_loader = load_mnist(datafolder, train_batch_size, train=True, download=False, shuffle=False)
input_test_loader = load_mnist(datafolder, 1, train=False, download=False, shuffle=False)

# model loading
if dist:
    fullmodelname = "distModel%s" % modelname
else:
    fullmodelname = "featsModel%s" % modelname
if dist:
    model = load_model(distanceModel, model_folder, fullmodelname, trainstep, pretrained)
else:
    if ae:
        model = load_model(featsLenetAE, model_folder, fullmodelname, trainstep, pretrained, outDim=outDim)
    else:
        model = load_model(featsLenet, model_folder, fullmodelname, trainstep, pretrained, outDim=outDim)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model.cuda()
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# writers and criterion
# writer = SummaryWriter(comment='%s_kNeighbours' % (log_name))

# Training
print('Start Full Search')
print(log_name)

total = 0
correct = 0
iterInputTestLoader = iter(input_test_loader)
for i in range(N_test_samples):
    input_test, label_test = next(iterInputTestLoader)
    input_test_batch = input_test.expand(train_batch_size, -1, -1, -1)
    bestDistance = torch.ones(1)*1e+5
    iterSearchTrainLoader = iter(search_train_loader)
    if torch.cuda.is_available():
        input_test_batch = input_test_batch.cuda()
    for j in range(N_train_batches):
        input_train_search, label_train_search = next(iterSearchTrainLoader)
        if torch.cuda.is_available():
            input_train_search = input_train_search.cuda()

        if dist:
            distancesTmp = model.forward(input_test_batch, input_train_search)
        else:
            if ae:
                output_test_batch = model.encoder(input_test_batch)
                output_train_search = model.encoder(input_train_search)
            else:
                output_test_batch = model.forward(input_test_batch)
                output_train_search = model.forward(input_train_search)
            distancesTmp = mse_batch(output_train_search-output_test_batch)

        sortedDistances, sortedIndices = distancesTmp.sort(0)
        bestDistanceTmp = sortedDistances[0]
        if bestDistanceTmp<bestDistance:
            bestDistance = bestDistanceTmp
            nnLabel = label_train_search[sortedIndices[0]]

    total += 1
    correct += (nnLabel == label_test[0]).sum().item()
    if i%100==0:
        print("Total: %i,   correct: %i, accuracy: %f %%" % (total, correct, 100 * correct / total))


print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))


print('Finished Full Search')
print(log_name)

# save distances?
# writer.close()