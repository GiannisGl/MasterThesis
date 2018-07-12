from trainModels.lenet import lenet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from LearnDistance.refactored.helperFunctions import *
from tensorboardX import SummaryWriter

case = "lenet5mnistNoNormAugmented"
model_folder = "models"

trainstep = 1
if torch.cuda.is_available():
    batch_size = 60000
else:
    batch_size = 100
Nepochs = 100
learningRate = 1e-3

if torch.cuda.is_available():
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../data"

# no normalization
transform = transforms.Compose(
    [transforms.RandomAffine(degrees=10, translate=[0.2,0.2], shear=5),
        transforms.ToTensor()])

data_train = MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)

data_test = MNIST(root=data_folder, train=False,
                                       download=False, transform=transform)

data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0)
data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=0)


modelname = "%s" % (case)
model = load_model(lenet5, model_folder, modelname, trainstep-1)
log_name = "%sBatch%iLR%f_Iter%i" % (modelname, batch_size, learningRate, trainstep)

writer = SummaryWriter(comment='%s_loss_log' % (log_name))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

def train(epoch):
    model.train()
    running_loss = 0.0
    total_correct = 0
    log_iter = 100
    for i, (images, labels) in enumerate(data_train_loader, 0):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            images = images.cuda()
            labels = labels.cuda()
            criterion.cuda()

        optimizer.zero_grad()

        output = model(images)

        if torch.cuda.is_available():
            output = output.cuda()

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()
        if i % log_iter == 0:
            writer.add_image(tag='image', img_tensor=images[1], global_step=Nepochs*60000+i)
            writer.add_scalar(tag='loss', scalar_value=loss, global_step=Nepochs*60000+i)
            print('[%d, %5d] loss: %f, Accuracy: %f' %
                  (epoch+1, i, running_loss/log_iter, float(total_correct)/log_iter))
            running_loss = 0.0
            total_correct = 0


def test():
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            images = images.cuda()
            labels = labels.cuda()
        output = model(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    print('Finished Epoch')
    test()
    save_model_weights(model, model_folder, modelname, trainstep)
    print('saved model')


def main():
    for e in range(0, Nepochs):
        train_and_test(e)

main()
