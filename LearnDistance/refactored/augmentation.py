import torch
from numpy.random.mtrand import rand, randn, randint
from scipy.ndimage import rotate


def random_augmentation(imageBatch):
    if(rand()>=0.1):
        imageBatch = random_rotate(imageBatch)
    if(rand()>=0.1):
        imageBatch = random_shift(imageBatch)
    if(rand()>=0.1):
        imageBatch = gaussian_noise(imageBatch)
    return imageBatch


def gaussian_noise(imageBatch):
    b,c,x,y = imageBatch.shape
    gaussian = randn(b,c,x,y)*0.05
    return imageBatch+torch.Tensor(gaussian)


def blur(imageBatch):
    return imageBatch


def random_shift(imageBatch):
    shiftRange = 5
    _, _, x, y = imageBatch.shape
    xShift = randint(-shiftRange,shiftRange)
    yShift = randint(-shiftRange,shiftRange)
    startX = abs(xShift)
    startY = abs(yShift)
    zeros = torch.zeros([sum(x) for x in zip(imageBatch.shape,(0,0,startX*2,startY*2))])
    zeros[:,:,startX:startX+x, startY:startY+y] = imageBatch[:,:,:,:]
    imageBatch_cropped = crop(zeros, startX-xShift, startY-yShift, imageBatch.shape)
    return torch.Tensor(imageBatch_cropped)


def random_rotate(imageBatch):
    angleRange = 15
    angle = randint(-angleRange,angleRange)
    if torch.cuda.is_available():
        imageBatch_rotated = crop_center(rotate(imageBatch, angle, axes=(2,3)), imageBatch.shape)
    else:
        imageBatch_rotated = crop_center(rotate(imageBatch.detach().numpy(), angle, axes=(2,3)), imageBatch.shape)
    return torch.Tensor(imageBatch_rotated)


def crop(imageBatch, startX, startY, newShape):
    _, _, x, y = imageBatch.shape
    _, _, newX, newY = newShape
    endX = min(startX+newX, x)
    endY = min(startY+newY, y)
    return imageBatch[:, :, startX:endX, startY:endY]


def crop_center(imageBatch,newShape):
    _,_,x,y = imageBatch.shape
    newB, newC, newX, newY = newShape
    startX = x//2 - newX//2
    startY = y//2 - newY//2
    return crop(imageBatch, startX, startY, newShape)
