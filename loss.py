from torch.nn import MSELoss
import torch


def distance_loss(input1, input2, output1, output2):
    distanceOrig = torch.mean(torch.pow(input1-input2,2))
    print("loss orig: ", distanceOrig)
    # distanceOrig = torch.nn.MSELoss(input1,input2)
    distanceOut = torch.mean(torch.pow(output1-output2,2))
    print("loss out: ", distanceOut)
    loss = torch.mean(distanceOrig-distanceOut)
    print("loss: ", loss)
    return loss
