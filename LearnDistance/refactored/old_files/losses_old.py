import torch
from helperFunctions import augment_batch


class distance_loss_part(torch.nn.Module):
    def __init__(self, writer, log_iter, delta, lamda, nAug=3, Aug=True):
        super(distance_loss_part, self).__init__()
        self.writer = writer
        self.log_iter = log_iter
        self.step = 0
        self.delta = delta
        self.lamda = lamda
        self.nAug = nAug
        self.Aug = Aug

    def forward(self, input1, input2, input3, featsModel, distanceModel):
        self.step += 1
        delta = torch.ones(input1.size()[0]) * self.delta
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # get features of inputs
        ## copy inputs
        input1feats = featsModel.forward(input1)
        input2feats = featsModel.forward(input2)
        input3feats = featsModel.forward(input3)

        # get L2 distance of the 3 pairs of features
        dist12 = mse_batch_loss(input1feats, input2feats)
        dist13 = mse_batch_loss(input1feats, input3feats)
        dist23 = mse_batch_loss(input2feats, input3feats)

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

        # Features model terms
        featsLoss = 0.0
        # terms that preserve distance
        featsLossDist = mseLoss(dist12, learnedDist12)
        featsLossDist += mseLoss(dist12, learnedDist21)
        featsLossDist += mseLoss(dist13, learnedDist13)
        featsLossDist += mseLoss(dist13, learnedDist31)
        featsLossDist += mseLoss(dist23, learnedDist23)
        featsLossDist += mseLoss(dist23, learnedDist32)
        self.writer.add_scalar(tag='featsLossDist', scalar_value=featsLossDist, global_step=self.step)
        featsLoss += featsLossDist

        # Distance model terms
        distLoss = 0.0

        # terms that enforce positivity
        distLossPos = mseLoss(relu(-learnedDist11))
        distLossPos += mseLoss(relu(-learnedDist22))
        distLossPos += mseLoss(relu(-learnedDist12))
        distLossPos += mseLoss(relu(-learnedDist21))
        distLossPos += mseLoss(relu(-learnedDist13))
        distLossPos += mseLoss(relu(-learnedDist31))
        distLossPos += mseLoss(relu(-learnedDist23))
        distLossPos += mseLoss(relu(-learnedDist32))
        self.writer.add_scalar(tag='distLossPos', scalar_value=distLossPos, global_step=self.step)
        distLoss += distLossPos

        # terms that enforce 0 distance for same inputs
        distLossId = mseLoss(learnedDist11)
        distLossId += mseLoss(learnedDist22)
        distLossId += mseLoss(learnedDist33)
        self.writer.add_scalar(tag='distLossId', scalar_value=distLossId, global_step=self.step)
        distLoss += distLossId

        # terms that enforce symmetry
        distLossSymm = mseLoss(learnedDist12, learnedDist21)
        distLossSymm += mseLoss(learnedDist13, learnedDist31)
        distLossSymm += mseLoss(learnedDist23, learnedDist32)
        self.writer.add_scalar(tag='distLossSymm', scalar_value=distLossSymm, global_step=self.step)
        distLoss += distLossSymm

        # # terms that enforce triangular inequality
        # distLossIneq = mseLoss(relu(learnedDist13 - learnedDist12 - learnedDist23))
        # distLossIneq += mseLoss(relu(learnedDist31 - learnedDist32 - learnedDist21))
        # distLossIneq += mseLoss(relu(learnedDist23 - learnedDist21 - learnedDist13))
        # distLossIneq += mseLoss(relu(learnedDist32 - learnedDist31 - learnedDist12))
        # distLossIneq += mseLoss(relu(learnedDist12 - learnedDist13 - learnedDist32))
        # distLossIneq += mseLoss(relu(learnedDist21 - learnedDist23 - learnedDist31))
        # self.writer.add_scalar(tag='distLossIneq', scalar_value=distLossIneq, global_step=self.step)
        # distLoss += distLossIneq
        # terms that enforce distance greater than delta

        distLossDelta = mseLoss(relu(delta - learnedDist12))
        distLossDelta += mseLoss(relu(delta - learnedDist13))
        distLossDelta += mseLoss(relu(delta - learnedDist23))
        distLossDelta += mseLoss(relu(delta - learnedDist21))
        distLossDelta += mseLoss(relu(delta - learnedDist32))
        distLossDelta += mseLoss(relu(delta - learnedDist31))
        self.writer.add_scalar(tag='distLossDelta', scalar_value=distLossDelta, global_step=self.step)
        distLoss += distLossDelta

        if self.Aug==True:
            # Augmentation terms
            featsLossClust = 0.0
            distLossNeigh = 0.0
            for i in range(self.nAug):
                input1augm = augment_batch(input1)
                input1augmfeats = featsModel.forward(input1augm)
                input2augm = augment_batch(input2)
                input2augmfeats = featsModel.forward(input2augm)
                input3augm = augment_batch(input3)
                input3augmfeats = featsModel.forward(input3augm)
                # terms that enforce clustering
                featsLossClust += mseLoss(input1feats, input1augmfeats)
                featsLossClust += mseLoss(input2feats, input2augmfeats)
                featsLossClust += mseLoss(input3feats, input3augmfeats)

                # get learned distance of input and its augmentation (should be zero)
                learnedDist11aug = distanceModel(input1, input1augm)
                learnedDist1aug1 = distanceModel(input1augm, input1)
                learnedDist22aug = distanceModel(input2, input2augm)
                learnedDist2aug2 = distanceModel(input2augm, input2)
                learnedDist33aug = distanceModel(input3, input3augm)
                learnedDist3aug3 = distanceModel(input3augm, input3)

                # terms that enforce neighbourhood
                distLossNeigh += mseLoss(learnedDist11aug)
                distLossNeigh += mseLoss(learnedDist22aug)
                distLossNeigh += mseLoss(learnedDist33aug)
                distLossNeigh += mseLoss(learnedDist1aug1)
                distLossNeigh += mseLoss(learnedDist2aug2)
                distLossNeigh += mseLoss(learnedDist3aug3)

            self.writer.add_scalar(tag='featsLossClust', scalar_value=featsLossClust, global_step=self.step)
            featsLoss += featsLossClust
            self.writer.add_scalar(tag='distLossNeigh', scalar_value=distLossNeigh, global_step=self.step)
            distLoss += distLossNeigh


        self.writer.add_scalar(tag='featsLoss', scalar_value=featsLoss, global_step=self.step)
        self.writer.add_scalar(tag='distLoss', scalar_value=distLoss, global_step=self.step)
        loss = featsLoss + self.lamda * distLoss
        self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=self.step)

        return loss


class distance_loss_convFeats(torch.nn.Module):
    def __init__(self, writer, delta, lamda, nAug=3):
        super(distance_loss, self).__init__()
        self.writer = writer
        self.step = 0
        self.delta = delta
        self.lamda = lamda
        self.nAug = nAug

    def forward(self, input1, input2, input3, featsModel, distanceModel):
        self.step += 1
        delta = torch.ones(input1.size()[0]) * self.delta
        zero = torch.zeros(input1.size()[0])
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # get features of inputs
        ## copy inputs
        input1feats = featsModel.convnet(input1)
        input1augm = augment_batch(input1)
        input1augmfeats = featsModel.convnet(input1augm)
        input2feats = featsModel.convnet(input2)
        input2augm = augment_batch(input2)
        input2augmfeats = featsModel.convnet(input2augm)
        input3feats = featsModel.convnet(input3)
        input3augm = augment_batch(input3)
        input3augmfeats = featsModel.convnet(input3augm)

        # get L2 distance of the 3 pairs of features
        dist12 = mse_batch_loss(input1feats, input2feats)
        dist13 = mse_batch_loss(input1feats, input3feats)
        dist23 = mse_batch_loss(input2feats, input3feats)

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

        # Features model terms
        featsLoss = 0
        # terms that preserve distance
        featsLossDist = mseLoss(dist12, learnedDist12)
        featsLossDist += mseLoss(dist13, learnedDist13)
        featsLossDist += mseLoss(dist23, learnedDist23)
        self.writer.add_scalar(tag='featsLossDist', scalar_value=featsLossDist, global_step=self.step)
        featsLoss += featsLossDist

        # Distance model terms
        distLoss = 0

        # terms that enforce positivity
        distLossPos = mseLoss(relu(-learnedDist11))
        distLossPos += mseLoss(relu(-learnedDist22))
        distLossPos += mseLoss(relu(-learnedDist12))
        distLossPos += mseLoss(relu(-learnedDist21))
        distLossPos += mseLoss(relu(-learnedDist13))
        distLossPos += mseLoss(relu(-learnedDist31))
        distLossPos += mseLoss(relu(-learnedDist23))
        distLossPos += mseLoss(relu(-learnedDist32))
        self.writer.add_scalar(tag='distLossPos', scalar_value=distLossPos, global_step=self.step)
        distLoss += distLossPos

        # terms that enforce 0 distance for same inputs
        distLossId = mseLoss(learnedDist11)
        distLossId += mseLoss(learnedDist22)
        distLossId += mseLoss(learnedDist33)
        self.writer.add_scalar(tag='distLossId', scalar_value=distLossId, global_step=self.step)
        distLoss += distLossId

        # terms that enforce symmetry
        distLossSymm = mseLoss(learnedDist12, learnedDist21)
        distLossSymm += mseLoss(learnedDist13, learnedDist31)
        distLossSymm += mseLoss(learnedDist23, learnedDist32)
        self.writer.add_scalar(tag='distLossSymm', scalar_value=distLossSymm, global_step=self.step)
        distLoss += distLossSymm

        # terms that enforce distance greater than delta
        distLossDelta = mseLoss(relu(delta - learnedDist12))
        distLossDelta += mseLoss(relu(delta - learnedDist13))
        distLossDelta += mseLoss(relu(delta - learnedDist23))
        distLossDelta += mseLoss(relu(delta - learnedDist21))
        distLossDelta += mseLoss(relu(delta - learnedDist32))
        distLossDelta += mseLoss(relu(delta - learnedDist31))
        self.writer.add_scalar(tag='distLossDelta', scalar_value=distLossDelta, global_step=self.step)
        distLoss += distLossDelta

        # terms that enforce triangular inequality
        distLossIneq = mseLoss(relu(learnedDist13 - learnedDist12 - learnedDist23))
        distLossIneq += mseLoss(relu(learnedDist31 - learnedDist32 - learnedDist21))
        distLossIneq += mseLoss(relu(learnedDist23 - learnedDist21 - learnedDist13))
        distLossIneq += mseLoss(relu(learnedDist32 - learnedDist31 - learnedDist12))
        distLossIneq += mseLoss(relu(learnedDist12 - learnedDist13 - learnedDist32))
        distLossIneq += mseLoss(relu(learnedDist21 - learnedDist23 - learnedDist31))
        self.writer.add_scalar(tag='distLossIneq', scalar_value=distLossIneq, global_step=self.step)
        distLoss += distLossIneq

        # Augmentation terms
        featsLossClust = 0.0
        distLossNeigh = 0.0
        for i in range(self.nAug):
            input1augm = augment_batch(input1)
            input1augmfeats = featsModel.forward(input1augm)
            input2augm = augment_batch(input2)
            input2augmfeats = featsModel.forward(input2augm)
            input3augm = augment_batch(input3)
            input3augmfeats = featsModel.forward(input3augm)
            # terms that enforce clustering
            featsLossClust += mseLoss(input1feats, input1augmfeats)
            featsLossClust += mseLoss(input2feats, input2augmfeats)
            featsLossClust += mseLoss(input3feats, input3augmfeats)

            # get learned distance of input and its augmentation (should be zero)
            learnedDist11aug = distanceModel(input1, input1augm)
            learnedDist22aug = distanceModel(input2, input2augm)
            learnedDist33aug = distanceModel(input3, input3augm)

            # terms that enforce neighbourhood
            distLossNeigh += mseLoss(learnedDist11aug)
            distLossNeigh += mseLoss(learnedDist22aug)
            distLossNeigh += mseLoss(learnedDist33aug)

        self.writer.add_scalar(tag='featsLossClust', scalar_value=featsLossClust, global_step=self.step)
        featsLoss += featsLossClust
        self.writer.add_scalar(tag='distLossNeigh', scalar_value=distLossNeigh, global_step=self.step)
        distLoss += distLossNeigh

        self.writer.add_scalar(tag='featsLoss', scalar_value=featsLoss, global_step=self.step)
        self.writer.add_scalar(tag='distLoss', scalar_value=distLoss, global_step=self.step)
        loss = featsLoss + self.lamda * distLoss
        self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=self.step)

        return loss


class distance_loss_fixFeatsConv(torch.nn.Module):
    def __init__(self, writer, delta, lamda):
        super(distance_loss_fixFeatsConv, self).__init__()
        self.writer = writer
        self.step = 0
        self.delta = delta
        self.lamda = lamda

    def forward(self, input1, input2, input3, featsModel, distanceModel):
        self.step += 1
        delta = torch.ones(input1.size()[0]) * self.delta
        zero = torch.zeros(input1.size()[0])
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # get features of inputs
        input1feats = featsModel.convnet(input1)
        input1augm = augment_batch(input1)
        input1augmfeats = featsModel.convnet(input1augm)
        input2feats = featsModel.convnet(input2)
        input2augm = augment_batch(input2)
        input2augmfeats = featsModel.convnet(input2augm)
        input3feats = featsModel.convnet(input3)
        input3augm = augment_batch(input3)
        input3augmfeats = featsModel.convnet(input3augm)

        # get L2 distance of the 3 pairs of features
        dist12 = mse_batch_loss(input1feats, input2feats)
        dist13 = mse_batch_loss(input1feats, input3feats)
        dist23 = mse_batch_loss(input2feats, input3feats)

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

        # Features model terms
        featsLoss = 0
        # terms that preserve distance
        featsLossDist = mseLoss(dist12, learnedDist12)
        featsLossDist += mseLoss(dist13, learnedDist13)
        featsLossDist += mseLoss(dist23, learnedDist23)
        self.writer.add_scalar(tag='featsLossDist', scalar_value=featsLossDist, global_step=self.step)
        featsLoss += featsLossDist
        # terms that enforce clustering
        featsLossClust = mseLoss(input1feats, input1augmfeats)
        featsLossClust += mseLoss(input2feats, input2augmfeats)
        featsLossClust += mseLoss(input3feats, input3augmfeats)
        self.writer.add_scalar(tag='featsLossClust', scalar_value=featsLossClust, global_step=self.step)
        featsLoss += featsLossClust

        self.writer.add_scalar(tag='featsLoss', scalar_value=featsLoss, global_step=self.step)

        # Distance model terms
        distLoss = 0

        #terms that enforce positivity
        distLossPos = mseLoss(relu(-learnedDist11))
        distLossPos += mseLoss(relu(-learnedDist22))
        distLossPos += mseLoss(relu(-learnedDist12))
        distLossPos += mseLoss(relu(-learnedDist21))
        distLossPos += mseLoss(relu(-learnedDist13))
        distLossPos += mseLoss(relu(-learnedDist31))
        distLossPos += mseLoss(relu(-learnedDist23))
        distLossPos += mseLoss(relu(-learnedDist32))
        self.writer.add_scalar(tag='distLossPos', scalar_value=distLossPos, global_step=self.step)
        distLoss += distLossPos

        # terms that enforce 0 distance for same inputs
        distLossId = mseLoss(learnedDist11)
        distLossId += mseLoss(learnedDist22)
        distLossId += mseLoss(learnedDist33)
        self.writer.add_scalar(tag='distLossId', scalar_value=distLossId, global_step=self.step)
        distLoss += distLossId

        # terms that enforce symmetry
        distLossSymm = mseLoss(learnedDist12, learnedDist21)
        distLossSymm += mseLoss(learnedDist13, learnedDist31)
        distLossSymm += mseLoss(learnedDist23, learnedDist32)
        self.writer.add_scalar(tag='distLossSymm', scalar_value=distLossSymm, global_step=self.step)
        distLoss += distLossSymm

        # terms that enforce distance greater than delta
        distLossDelta = mseLoss(relu(delta - learnedDist12))
        distLossDelta += mseLoss(relu(delta - learnedDist13))
        distLossDelta += mseLoss(relu(delta - learnedDist23))
        distLossDelta += mseLoss(relu(delta - learnedDist21))
        distLossDelta += mseLoss(relu(delta - learnedDist32))
        distLossDelta += mseLoss(relu(delta - learnedDist31))
        self.writer.add_scalar(tag='distLossDelta', scalar_value=distLossDelta, global_step=self.step)
        distLoss += distLossDelta

        # terms that enforce triangular inequality
        distLossIneq = mseLoss(relu(learnedDist13 - learnedDist12 - learnedDist23))
        distLossIneq += mseLoss(relu(learnedDist31 - learnedDist32 - learnedDist21))
        distLossIneq += mseLoss(relu(learnedDist23 - learnedDist21 - learnedDist13))
        distLossIneq += mseLoss(relu(learnedDist32 - learnedDist31 - learnedDist12))
        distLossIneq += mseLoss(relu(learnedDist12 - learnedDist13 - learnedDist32))
        distLossIneq += mseLoss(relu(learnedDist21 - learnedDist23 - learnedDist31))
        self.writer.add_scalar(tag='distLossIneq', scalar_value=distLossIneq, global_step=self.step)
        distLoss += distLossIneq

        self.writer.add_scalar(tag='distLoss', scalar_value=distLoss, global_step=self.step)

        loss = featsLoss + self.lamda * distLoss
        self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=self.step)

        return loss


def mse_batch(input):
    return torch.mean(torch.pow(flatten(input),2),1)

def mseLoss(input, target=None):
    if target is None:
        target = torch.zeros(input.size())
    return torch.mean(torch.pow(flatten(input)-flatten(target),2))

def mse_batch_loss(input, target):
    return mse_batch(flatten(input) - flatten(target))

def flatten(input):
    return input.view(input.size()[0],-1)

def relu(input):
    return torch.clamp(input, min=0)
