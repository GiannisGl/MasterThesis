def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(1.0, 0.02)
    elif classname.find('Linear') != -1:
        module.weight.data.normal_(2.0, 0.02)
        module.bias.data.fill_(1)

def model_weights_init(model):
    for module in model.modules():
        weights_init(module)