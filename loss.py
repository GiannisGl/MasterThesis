from torch.nn import MSELoss
import torch


def distance_loss(input1, input2, output1, output2):
    distanceOrig = mse_loss(input1, input2)
    print("loss orig: ", distanceOrig)
    # distanceOrig = torch.nn.MSELoss(input1,input2)
    distanceOut = mse_loss(output1, output2)
    print("loss out: ", distanceOut)
    loss = torch.mean(distanceOrig-distanceOut)
    print("loss: ", loss)
    return loss

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()