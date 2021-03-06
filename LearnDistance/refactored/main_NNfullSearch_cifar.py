import torch
from inceptionModel import distInception
from helperFunctions import *
from losses import *

# Nearest Neighbor search for CIFAR10

# parameters and names
case = "CifarRelaxedWithClustering"
outDim = 3
nAug = 5
delta = 5
trainstep = 3
dataset = 'cifar'
# Per Epoch one iteration over the dataset
N_test_samples = 10000
dataset_size = 50000
if torch.cuda.is_available():
    train_batch_size = 800
    N_train_batches = int(dataset_size/train_batch_size)
    datafolder = "/var/tmp/ioannis/data"
else:
    train_batch_size = 100
    N_train_batches = 20
    N_test_samples = 1
    datafolder = "../../data"

lamda = 1
distPretrained = False
if case == "Autoencoder":
    modelname = "DistInception%sOut%iDelta%iLamda%i" % (case, outDim, delta, lamda)
    ae = True
else:
    modelname = "DistInception%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
    ae = False
log_name = "fullSearch%s%s_Iter%i" % (dataset, modelname, trainstep)
model_folder = "trainedModels"

search_train_loader = load_cifar(datafolder, train_batch_size, train=True, download=True, shuffle=False)
input_test_loader = load_cifar(datafolder, 1, train=False, download=True, shuffle=False)

# model loading
distModelname = "distModel%s" % modelname
distModel = load_model(distInception, model_folder, distModelname, trainstep, distPretrained)
distModel = distModel.eval()
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    distModel.cuda()
else:
    torch.set_default_tensor_type('torch.FloatTensor')


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

        distancesTmp = distModel.forward(input_test_batch, input_train_search)
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