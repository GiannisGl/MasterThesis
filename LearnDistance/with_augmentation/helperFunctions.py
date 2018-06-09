# from torch import im


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        module.weight.data.normal_(0.0, 0.1)
        module.bias.data.fill_(0)


def model_weights_init(model):
    for module in model.modules():
        weights_init(module)


