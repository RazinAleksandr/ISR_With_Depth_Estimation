import os
from typing import Optional, Any
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
    

class ImageDataset(Dataset):
    def __init__(self, image_dir: str, transform: Optional[Any] = None):
        self.image_dir = image_dir
        self.transform = transform
        self.file_names = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.file_names[idx] + '.jpg')
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)
        
        return image

class ConditionDataset(Dataset):
    def __init__(self, image_dir: str, depth_dir: str, transform: Optional[Any] = None, degradation: Optional[Any] = None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir

        self.transform = transform
        self.degradation = degradation
        
        self.file_names = sorted([os.path.splitext(f)[0] for f in os.listdir(depth_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.file_names[idx] + '.jpg')
        depth_name = os.path.join(self.depth_dir, self.file_names[idx] + '.png')
        
        image = Image.open(img_name)
        depth_map = Image.open(depth_name)
        
        if self.degradation:
            degraded_image = self.degradation(image)
        else:
            degraded_image = image

        if self.transform:
            degraded_image = self.transform(degraded_image)
            depth_map = self.transform(depth_map)
        else:
            degraded_image = ToTensor()(degraded_image)
            depth_map = ToTensor()(depth_map)
        
        return degraded_image, depth_map

