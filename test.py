import os
import numpy as np
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),

])
train_dataset = datasets.ImageFolder(
    root='dataset/spec',
    transform=train_transform
)
test_dataset = datasets.ImageFolder(
    root='dataset/spec_test',
    transform=train_transform
)

print(type(train_dataset.imgs))