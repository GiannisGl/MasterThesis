import torch
import torchvision
import torchvision.transforms as transforms


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
        module.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        module.weight.data.normal_(0.0, 0.1)
        module.bias.data.fill_(0)


def model_weights_random_gaussian(model):
    for module in model.modules():
        weights_init(module)


def load_model_weights(model, modelfilename):
    modelfile = torch.load(modelfilename)
    model.load_state_dict(modelfile)
    return model


def load_model(modelFunction, model_folder, modelname, trainstep, pretrained=False):
    model = modelFunction(pretrained)
    if trainstep<1 & (not pretrained):
        model_weights_random_gaussian(model)
    elif trainstep>=1:
        print("load")
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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    set = torchvision.datasets.MNIST(root=data_folder, train=train, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader
