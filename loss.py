from torch.nn import MSELoss
import torch

class distance_loss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(distance_loss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, output1, output2):
        distanceOrig = mse_loss(flatten(input1), flatten(input2))
        print("loss orig: ", distanceOrig)
        # distanceOrig = torch.nn.MSELoss(input1,input2)
        distanceOut = mse_loss(flatten(output1), flatten(output2))
        print("loss out: ", distanceOut)
        loss = torch.mean(torch.abs(distanceOrig - distanceOut))
        print("loss: ", loss)

        return loss

def mse_loss(input, target):
    return torch.mean(torch.pow(input - target,2),1)

def flatten(input):
    return input.view(input.size()[0],-1)