import torch


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


def load_model(modelFunction, pretrained, trainstep, model_folder, modelname):
    model = modelFunction(pretrained)
    if trainstep<=1 & (not pretrained):
        weights_init(model)
    elif trainstep>1:
        modelfilename = '%s/%s_Iter%i.state' % (model_folder, modelname, trainstep)
        load_model_weights(model,modelfilename)


def save_model_weights(model, model_folder, modelname, trainstep):
    modelfilename = '%s/%s_Iter%i.state' % (model_folder, modelname, trainstep)
    modelfile = open(modelfilename, "wb")
    torch.save(model.state_dict(), modelfile)