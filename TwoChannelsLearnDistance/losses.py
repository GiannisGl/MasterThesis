from torch.nn import MSELoss
import torch

class distance_loss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(distance_loss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, input3, featsModel, distanceModel):

        loss = 0
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # get features of inputs
        input1feats = featsModel.forward(input1)
        input2feats = featsModel.forward(input2)
        input3feats = featsModel.forward(input3)

        # get L2 distance of the 3 pairs of features
        dist12 = mse_loss(input1feats, input2feats)
        dist13 = mse_loss(input1feats, input3feats)
        dist23 = mse_loss(input2feats, input3feats)

        # get learned distance of the 3 pairs both ways
        learnedDist12 = distanceModel(input1, input2)
        learnedDist21 = distanceModel(input2, input1)
        learnedDist13 = distanceModel(input1, input3)
        learnedDist31 = distanceModel(input3, input1)
        learnedDist23 = distanceModel(input2, input3)
        learnedDist32 = distanceModel(input3, input2)

        # terms that preserve distance
        loss += mse_loss(dist12, learnedDist12)
        loss += mse_loss(dist13, learnedDist13)
        loss += mse_loss(dist23, learnedDist23)
        # terms that enforce symmetry
        loss += mse_loss(learnedDist12, learnedDist21)
        loss += mse_loss(learnedDist13, learnedDist31)
        loss += mse_loss(learnedDist23, learnedDist32)
        # terms that enforce equality



        # terms that enforce triangular inequality
        loss += math.max(learnedDist13-learnedDist12-learnedDist23,0)
        loss += math.max(learnedDist31-learnedDist32-learnedDist21,0)
        loss += math.max(learnedDist23-learnedDist21-learnedDist13,0)
        loss += math.max(learnedDist32-learnedDist31-learnedDist12,0)
        loss += math.max(learnedDist12-learnedDist13-learnedDist32,0)
        loss += math.max(learnedDist21-learnedDist23-learnedDist31,0)

        return loss

def mse_loss(input, target):
    return torch.mean(torch.pow(flatten(input) - flatten(target),2),1)

def flatten(input):
    return input.view(input.size()[0],-1)

def zeroOrDeltaDistance(sameInputs, delta, learnedDist):
    if(sameInputs):
        return 0
    else:
        return
