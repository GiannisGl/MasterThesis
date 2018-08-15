import torch
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


def freeze_first_conv_layers(model):
    i = 0
    for module in model.modules():
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            for param in module.parameters():
                param.requires_grad = False


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(module.weight)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(module.weight)
        init.constant_(module.bias, 0.1)

# torch.nn.init.xavier.normal


def model_weights_random_xavier(model):
    for module in model.modules():
        weights_init(module)


def load_model_weights(model, modelfilename):
    if torch.cuda.is_available():
        modelfile = torch.load(modelfilename)
    else:
        modelfile = torch.load(modelfilename, map_location=lambda storage, loc: storage)
    model.load_state_dict(modelfile)
    return model


def load_model(modelFunction, model_folder, modelname, trainstep, pretrained=False, outDim=None):
    if outDim is None:
        model = modelFunction(pretrained)
    else:
        model = modelFunction(outDim=outDim, pretrained=pretrained)
    if trainstep>=1:
        modelfilename = '%s/%s_Iter%i.state' % (model_folder, modelname, trainstep)
        load_model_weights(model,modelfilename)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def save_model_weights(model, model_folder, modelname, trainstep):
    modelfilename = '%s/%s_Iter%i.state' % (model_folder, modelname, trainstep)
    modelfile = open(modelfilename, "wb")
    torch.save(model.state_dict(), modelfile)


def load_mnist(data_folder, batch_size, train=True, download=False):
    # don't normalize
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root=data_folder, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def load_cifar(data_folder, batch_size, train=True, download=False):
    # don't normalize
    transform = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root=data_folder, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def augment_batch(batch):
    transformAug = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomAffine(scale=[0.8, 1.1], degrees=10, translate=[0.2, 0.2], shear=10),
         transforms.ToTensor()])
    batchSize = batch.shape[0]
    batchAug = torch.Tensor(batch.shape)
    for i in range(batchSize):
        batchAug[i] = transformAug(batch[i].cpu())
    return batchAug

# xflip, color jittering,


def initialize_pretrained_model(model, pretrained_filename):
    if torch.cuda.is_available():
        pretrained = torch.load(pretrained_filename)
    else:
        pretrained = torch.load(pretrained_filename, map_location=lambda storage, loc: storage)
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def visualize(writerEmb, model, datafolder, dataset='mnist', Nsamples=2000, train=True):
    if dataset=='mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(root=datafolder, train=train, download=False, transform=transform)
    elif dataset=='cifar':
        transform = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root=datafolder, train=train, download=False, transform=transform)
    subset = torch.utils.data.dataset.Subset(dataset, range(Nsamples))
    loader = torch.utils.data.DataLoader(subset, batch_size=Nsamples, shuffle=False, num_workers=0)

    iterLoader = iter(loader)
    input, label = next(iterLoader)
    if torch.cuda.is_available():
        output = model.forward(input.cuda())
    else:
        output = model.forward(input)
    output = torch.squeeze(output)
    if train:
        print('train: %s' % list(output.size()))
        writerEmb.add_embedding(output, label_img= input, metadata=label.numpy(), tag="1.train")
    else:
        print('test: %s' % list(output.size()))
        writerEmb.add_embedding(output, label_img= input, metadata=label.numpy(), tag="2.test")


