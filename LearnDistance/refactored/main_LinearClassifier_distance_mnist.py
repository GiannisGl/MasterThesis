import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from distanceModel import distanceModel
from helperFunctions import *
from losses import *


# parameters and names
case = "MnistExactWithClustering"
outDim = 3
nAug = 3
delta = 5
# trainstep of the trained model
trainstep = 2
# trainstep of the linear classifier
transferTrainstep = 0
learningRate = 1e-1
dataset = 'mnist'
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    train_batch_size = 1000
    Nsamples = int(60000 / train_batch_size)
    log_iter = int(Nsamples/2)
    Nepochs = 50
    datafolder = "/var/tmp/ioannis/data"
else:
    train_batch_size = 50
    Nsamples = int(60000 / train_batch_size)
    log_iter = int(Nsamples/2)
    Nepochs = 50
    datafolder = "../../data"

lamda = 1
distPretrained = False
modelname = "DistLeNet%sAug%iOut%iDelta%iLamda%i" % (case, nAug, outDim, delta, lamda)
log_name = "distTransfer%s%sAug%iBatch%iLR%f_Iter%i_Iter%i" % (dataset, modelname, nAug, train_batch_size, learningRate, trainstep, transferTrainstep)
model_folder = "trainedModels"

train_loader = load_mnist(datafolder, train_batch_size, train=True, download=True)

# model loading
distModelname = "distModel%s" % modelname
distModel = load_model(distanceModel, model_folder, distModelname, 0, distPretrained)
if transferTrainstep<1:
    distModel = load_model(distanceModel, model_folder, distModelname, trainstep, distPretrained)
freeze_layers(distModel)
# remove last layer
nClasses = 10
nFeats = distModel.classifier[-1].in_features
distModel.classifier[-1] = torch.nn.Linear(nFeats, nClasses)
print(distModel)
if transferTrainstep>=1:
    modelfilename = '%s/%sTransfer%s_Iter%i_Iter%i.state' % (model_folder, dataset, distModelname, trainstep, transferTrainstep)
    distModel = load_model_weights(distModel, modelfilename)
if torch.cuda.is_available():
    distModel.cuda()

# optimizers
distOptimizer = optim.Adam(distModel.classifier[-1].parameters(), lr=learningRate)
criterion = torch.nn.CrossEntropyLoss()

# writers and criterion
writer = SummaryWriter(comment='%s_loss_log' % log_name)

# Training
print('Start Training')
print(log_name)
for epoch in range(Nepochs):

    running_loss = 0.0
    iterTrainLoader = iter(train_loader)
    for i in range(Nsamples):
        input, label = next(iterTrainLoader)
        inputAug = augment_batch(input, dataset)

        # transfer to cuda if available
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            input = input.cuda()
            inputAug = inputAug.cuda()
            label = label.cuda()
            criterion.cuda()

        # zero the parameter gradients
        distOptimizer.zero_grad()

        # optimize
        output = distModel.forward(input, inputAug)
        loss = criterion(output, label)
        loss.backward()
        distOptimizer.step()
        global_step = epoch*Nsamples+i
        writer.add_scalar(tag='linearClassifier_dist_mnist', scalar_value=loss, global_step=global_step)

        # print statistics
        running_loss += loss.item()
        if i % log_iter == log_iter-1:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')
print(log_name)

# save weights
transferModelname = "%sTransfer%s_Iter%i" % (dataset, distModelname, trainstep)
print(transferModelname)
save_model_weights(distModel, model_folder, transferModelname, transferTrainstep + 1)
print('saved models')

writer.close()

test_loader = load_mnist(datafolder, train_batch_size, train=False, download=False)
test_accuracy(distModel, test_loader, dist=True)