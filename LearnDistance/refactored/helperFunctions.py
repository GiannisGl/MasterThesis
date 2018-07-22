import torch
import torch.nn.init as init
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
        init.xavier_normal_(module.weight)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(module.weight)
        init.constant_(module.bias, 0.1)

# torch.nn.init.xavier.normal


def model_weights_random_xavier(model):
    for module in model.modules():
        weights_init(module)


def load_model_weights(model, modelfilename):
    modelfile = torch.load(modelfilename)
    model.load_state_dict(modelfile)
    return model


def load_model(modelFunction, model_folder, modelname, trainstep, pretrained=False):
    model = modelFunction(pretrained)
    if trainstep<1 & (not pretrained):
        model_weights_random_xavier(model)
    elif trainstep>=1:
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


def augment_batch(batch):
    transformAug = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomAffine(scale=[0.8, 1.1], degrees=10, translate=[0.2, 0.2], shear=10),
         transforms.ToTensor()])
    batchSize = batch.shape[0]
    batchAug = torch.Tensor(batch.shape)
    for i in range(batchSize):
        batchAug[i] = transformAug(batch[i])
    return batchAug


def initialize_pretrained_model(model, pretrained_filename):
    pretrained = torch.load(pretrained_filename)
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
