import torch
from helperFunctions import *

class distance_loss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(distance_loss, self).__init__()
        self.margin = margin

    def forward(self, delta, input1, input2, input3, featsModel, distanceModel):
        delta = torch.ones(input1.size()[0])*delta
        zero = torch.zeros(input1.size()[0])
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # get features of inputs
        input1feats = featsModel.forward(input1)
        input1augm = random_augmentation(input1)
        input1augmfeats = featsModel.forward(input1augm)
        input2feats = featsModel.forward(input2)
        input2augm = random_augmentation(input2)
        input2augmfeats = featsModel.forward(input2augm)
        input3feats = featsModel.forward(input3)
        input3augm = random_augmentation(input3)
        input3augmfeats = featsModel.forward(input3augm)

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
        learnedDist11 = distanceModel(input1, input1)
        learnedDist22 = distanceModel(input2, input2)
        learnedDist33 = distanceModel(input3, input3)

        # terms that preserve distance
        featsLoss = mseLoss(dist12, learnedDist12)
        featsLoss += mseLoss(dist13, learnedDist13)
        featsLoss += mseLoss(dist23, learnedDist23)
        # terms that enforce clustering
        featsLoss += mse_loss(input1feats, input1augmfeats)
        featsLoss += mse_loss(input2feats, input2augmfeats)
        featsLoss += mse_loss(input3feats, input3augmfeats)

        # terms that enforce 0 distance for same inputs
        distLoss = mseLoss(learnedDist11, zero)
        distLoss += mseLoss(learnedDist22, zero)
        distLoss += mseLoss(learnedDist33, zero)

        # terms that enforce symmetry
        distLoss += mseLoss(learnedDist12, learnedDist21)
        distLoss += mseLoss(learnedDist13, learnedDist31)
        distLoss += mseLoss(learnedDist23, learnedDist32)

        # terms that enforce distance greater than delta
        distLoss += mseLoss(relu(delta - learnedDist12), zero)
        distLoss += mseLoss(relu(delta - learnedDist13), zero)
        distLoss += mseLoss(relu(delta - learnedDist23), zero)
        distLoss += mseLoss(relu(delta - learnedDist21), zero)
        distLoss += mseLoss(relu(delta - learnedDist32), zero)
        distLoss += mseLoss(relu(delta - learnedDist31), zero)

        # terms that enforce triangular inequality
        distLoss += mseLoss(relu(learnedDist13 - learnedDist12 - learnedDist23), zero)
        distLoss += mseLoss(relu(learnedDist31 - learnedDist32 - learnedDist21), zero)
        distLoss += mseLoss(relu(learnedDist23 - learnedDist21 - learnedDist13), zero)
        distLoss += mseLoss(relu(learnedDist32 - learnedDist31 - learnedDist12), zero)
        distLoss += mseLoss(relu(learnedDist12 - learnedDist13 - learnedDist32), zero)
        distLoss += mseLoss(relu(learnedDist21 - learnedDist23 - learnedDist31), zero)

        loss = [featsLoss, distLoss]

        return loss

def mse(input):
    return torch.mean(torch.pow(flatten(input),2),1)

def mseLoss(input, target):
    return torch.mean(torch.pow(flatten(input)-flatten(target),2))

def mse_loss(input, target):
    return mse(flatten(input) - flatten(target))

def flatten(input):
    #if(input.size().len()<=0):
     #   return input
    return input.view(input.size()[0],-1)

def relu(input):
    return torch.clamp(input, max=0)

def mse_relu(input):
    return mse(relu(input))
