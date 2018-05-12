from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import visdom

# viz = visdom.Visdom()

name = "lenet5"
model_folder = "models"

trainstep = 1
batch_size = 100
Nepochs = 1

if torch.cuda.is_available():
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../data"


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((-1, -1, -1), (1, 1, 1))])

data_train = MNIST(root=data_folder, train=True,
                                       download=False, transform=transform)

data_test = MNIST(root=data_folder, train=False,
                                       download=True, transform=transform)

data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)


if trainstep == 1:
    model = LeNet5()
else:
    modelfilename = '%s/lenetModel%s_Iter%i.torchmodel' % (model_folder, name, trainstep - 1)
    modelfile = open(modelfilename, 'rb')
    model = torch.load(modelfile)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)

if torch.cuda.is_available():
    model.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    criterion.cuda()


# cur_batch_win = None
# cur_batch_win_opts = {
#     'title': 'Epoch Loss Trace',
#     'xlabel': 'Batch Number',
#     'ylabel': 'Loss',
#     'width': 1200,
#     'height': 600,
# }


def train(epoch):
    # global cur_batch_win
    model.train()
    loss_list, batch_list = [], []
    running_loss = 0.0
    log_iter = 100
    for i, (images, labels) in enumerate(data_train_loader):
        # images, labels = Variable(images), Variable(labels)

        if torch.cuda.is_available():
            images.cuda()

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        # loss_list.append(loss.item())
        # batch_list.append(i+1)

        # # Update Visualization
        # if viz.check_connection():
        #     cur_batch_win = viz.line(torch.FloatTensor(loss_list), torch.FloatTensor(batch_list),
        #                              win=cur_batch_win, name='current_batch_loss',
        #                              update=(None if cur_batch_win is None else 'replace'),
        #                              opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % log_iter == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch+1, i, running_loss / log_iter))
            running_loss = 0.0


def test():
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        # images, labels = Variable(images), Variable(labels)
        if torch.cuda.is_available():
            images.cuda()
        output = model(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data[0], float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    print('Finished Training')
    test()
    modelfilename = '%s/lenetModel%s_Iter%i.torchmodel' % (model_folder, name, trainstep)
    modelfile = open(modelfilename, "wb")
    torch.save(model, modelfile)
    print('saved models')


def main():
    for e in range(0, Nepochs):
        train_and_test(e)


# if __name__ == '__main__':

main()