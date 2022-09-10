from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from parameters import NYUD_MEAN, NYUD_STD


class MyDataset(Dataset):
    def __init__(self, data, transform=False):
        self.data_paths = data
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, depth_path = self.data_paths[index]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(image)
        depth = convert_tensor(depth)

        if self.transform:
            if random.random() > 0.5:
                image = TF.hflip(image)
                depth = TF.hflip(depth)

            if random.random() > 0.5:
                image = TF.vflip(image)
                depth = TF.vflip(depth)

            transform = transforms.Normalize(NYUD_MEAN, NYUD_STD)
            image = transform(image)

        return image, depth

