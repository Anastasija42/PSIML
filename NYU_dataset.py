
import os
from tracemalloc import is_tracing
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


IMAGES_PATH = 'C:/Users/psiml8/Documents/GitHub/PSIML/npy/images.npy'
DEPTHS_PATH = 'C:/Users/psiml8/Documents/GitHub/PSIML/npy/depths.npy'

NYUD_MEAN = [0.485, 0.456, 0.406]
NYUD_STD = [0.229, 0.224, 0.225]

class MyDataset(Dataset):
    def __init__(self, data, transform = None):
        self.transform = transform
        self.images = data[:,0]
        self.depths = data[:,1]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        depth = self.depths[index]
        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).float() 

        return image, depth

class MyDataset2(Dataset):
    def __init__(self, data, is_training):
        self.data = data
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]["filename"]
        depth = self.data[index]["depth_map"]
        
        convert_tensor = transforms.ToTensor()
        image = Image.open(image)
        image =  convert_tensor(image)
        depth = Image.open(depth)
        depth =  convert_tensor(depth)

        if self.is_training:
            #transform = transforms.Normalize(NYUD_MEAN, NYUD_STD)
            #image = transform(image)
            #depth = transform(depth)

            if random.random() > 0.5:
                image = TF.hflip(image)
                depth = TF.hflip(depth)

            if random.random() > 0.5:
                image = TF.vflip(image)
                depth = TF.vflip(depth)
        else:
            #transform = transforms.Normalize(NYUD_MEAN, NYUD_STD)
            #image = transform(image)
            #depth = transform(depth)
            pass

        return image, depth

