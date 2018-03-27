from torch.nn import MSELoss
import torch

class distance_loss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(distance_loss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, output1, output2):
        distanceOrig = mse_loss(input1, input2)
        print("loss orig: ", distanceOrig)
        # distanceOrig = torch.nn.MSELoss(input1,input2)
        distanceOut = mse_loss(output1, output2)
        print("loss out: ", distanceOut)
        loss = mse_loss(distanceOrig, distanceOut)
        print("loss: ", loss)

        return loss

def mse_loss(input, target):
    return torch.sum(torch.pow(input - target,2)) / input.data.nelement()