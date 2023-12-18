import os

import numpy as np
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/spec"
moudle0_dir = "moudle0"
moudle1_dir = "moudle1"
moudle2_dir = "moudle2"
moudle3_dir = "moudle3"
moudle4_dir = "moudle4"
moudle5_dir = "moudle5"
moudle6_dir = "moudle6"
moudle7_dir = "moudle7"
moudle8_dir = "moudle8"
moudle9_dir = "moudle9"
moudle10_dir = "moudle10"

moudle0_dataset = MyData(root_dir, moudle0_dir)
moudle1_dataset = MyData(root_dir, moudle1_dir)
moudle2_dataset = MyData(root_dir, moudle2_dir)
moudle3_dataset = MyData(root_dir, moudle3_dir)
moudle4_dataset = MyData(root_dir, moudle4_dir)
moudle5_dataset = MyData(root_dir, moudle5_dir)
moudle6_dataset = MyData(root_dir, moudle6_dir)
moudle7_dataset = MyData(root_dir, moudle7_dir)
moudle8_dataset = MyData(root_dir, moudle8_dir)
moudle9_dataset = MyData(root_dir, moudle9_dir)
moudle10_dataset = MyData(root_dir, moudle10_dir)
train_dataset = moudle0_dataset+moudle1_dataset+moudle2_dataset+moudle3_dataset+moudle4_dataset+moudle5_dataset+moudle6_dataset+moudle7_dataset+moudle8_dataset+moudle9_dataset+moudle10_dataset
img,lable = train_dataset[50]

print(img)

resize = torchvision.transforms.Resize((100,100))
train_dataset = resize(train_dataset)
totensor = torchvision.transforms.ToTensor()
train_dataset = totensor(train_dataset)
img,lable = train_dataset[51]
print(img)