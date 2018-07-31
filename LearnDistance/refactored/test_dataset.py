import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


if torch.cuda.is_available():
    train_batch_size = 1
else:
    train_batch_size = 1
Nsamples = 1000

# dataset loading
if torch.cuda.is_available():
    data_folder = "/var/tmp/ioannis/data"
else:
    data_folder = "../../data"

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root=data_folder, train=True, download=False, transform=transform)
sampler = SubsetRandomSampler(range(Nsamples))
loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, sampler=sampler, shuffle=False, num_workers=0)

# Training
iterLoader = iter(loader)
counts = [0,0,0,0,0,0,0,0,0,0]
for i in range(Nsamples):
    input, label = next(iterLoader)
    log = "%i, %s" % (i, label.item())
    print(log)
    counts[label] += 1

print(counts)
