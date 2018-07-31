import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from augmentation import *
from featuresModel import featsLenet, featsLenetFull, featsLenetFix
from distanceModel import distanceModel
from helperFunctions import *
from losses import *
from tensorboardX import SummaryWriter


# parameters and names
case = "partFixFeats"
trainstep = 1
nAug = 5
# Per Epoch one iteration over the dataset
if torch.cuda.is_available():
    train_batch_size = 300
    Nepochs = 50
else:
    train_batch_size = 50
    Nepochs = 1
Nsamples = 900
Nbatches = Nsamples/(train_batch_size*3)
learningRate = 1e-3
delta = 5
lamda = 1
log_iter = int(Nsamples/2)
featsPretrained = True
distPretrained = False
modelname = "LearnDistanceDistLeNetNoNorm%sDelta%iLamda%i" % (case, delta, lamda)
log_name = "%sBatch%iLR%f_Iter%i" % (modelname, train_batch_size, learningRate, trainstep)
model_folder = "trainedModels"

# dataset loading
if torch.cuda.is_available():
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../../data"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root=data_folder, train=True, download=False, transform=transform)
train_sampler = SubsetRandomSampler(range(Nsamples))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, shuffle=False, num_workers=0)

# model loading
featsModelname = "featsModel%s" % modelname
featsModel = load_model(featsLenetFix, model_folder, featsModelname, 0, featsPretrained)
distModelname = "distModel%s" % modelname
distModel = load_model(distanceModel, model_folder, distModelname, trainstep-1, distPretrained)

# optimizers
distOptimizer = optim.Adam(distModel.parameters(), lr=learningRate)

# writer and criterion
writer = SummaryWriter(comment='%s_loss_log' % (log_name))
writer_img = SummaryWriter(comment='%s_images' % (log_name))
criterion = distance_loss_part(writer, writer_img, log_iter, delta, lamda, nAug)

# Training
print('Start Training')
print(log_name)
for epoch in range(Nepochs):

    running_loss = 0.0
    iterTrainLoader = iter(train_loader)
    for i in range(Nbatches):
        input1, _ = next(iterTrainLoader)
        input2, _ = next(iterTrainLoader)
        input3, _ = next(iterTrainLoader)

        # transfer to cuda if available
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            input1 = input1.cuda()
            input2 = input2.cuda()
            input3 = input3.cuda()
            criterion.cuda()

        # zero the parameter gradients
        distOptimizer.zero_grad()

        # optimize
        loss = criterion(input1, input2, input3, featsModel, distModel)
        loss.backward()
        distOptimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == log_iter-1:
            print('[%d, %5d] loss: %f' %
                  (epoch + 1, i, running_loss / log_iter))
            running_loss = 0.0

print('Finished Training')

writer.close()
writer_img.close()

# save weights
save_model_weights(distModel, model_folder, distModelname, trainstep)
print('saved models')


