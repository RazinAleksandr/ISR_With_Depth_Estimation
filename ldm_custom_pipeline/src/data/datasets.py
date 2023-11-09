# Standard libraries
import os

# Third-party libraries
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

# Local application/modules
from src.constants.types import Optional, Any, Tuple
from src.constants.constants import NUM_TRAIN_SAMPLES


class ImageDataset(Dataset):
    """
    Dataset class to handle image data.
    
    :param image_dir: Directory containing image data.
    :param transform: Transformation to be applied on the images.
    """
    def __init__(self, image_dir: str, transform: Optional[Any] = None):
        self.image_dir = image_dir
        self.transform = transform
        self.file_names = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.file_names = self.file_names[:NUM_TRAIN_SAMPLES]

    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx: int) -> Any:
        img_name = os.path.join(self.image_dir, self.file_names[idx] + '.jpg')
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)
        
        return image


class ConditionDataset(Dataset):
    """
    Dataset class to handle image and corresponding depth data.
    
    :param image_dir: Directory containing image data.
    :param depth_dir: Directory containing depth data.
    :param transform: Transformation to be applied on the images and depth maps.
    :param degradation: Degradation function to be applied on the images.
    """
    def __init__(self, image_dir: str, depth_dir: str, transform: Optional[Any] = None, degradation: Optional[Any] = None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.degradation = degradation
        
        self.file_names = sorted([os.path.splitext(f)[0] for f in os.listdir(depth_dir) if f.endswith('.png')])
        self.file_names = self.file_names[:NUM_TRAIN_SAMPLES]

    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
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


class CombinedDataset(Dataset):
    """
    Dataset class to combine Image and Condition datasets.
    
    :param image_dataset: An instance of the ImageDataset.
    :param cond_dataset: An instance of the ConditionDataset.
    """
    def __init__(self, image_dataset: Dataset, cond_dataset: Dataset):
        self.image_dataset = image_dataset
        self.cond_dataset = cond_dataset
        assert len(self.image_dataset) == len(self.cond_dataset), "Datasets must be of the same size"

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Tuple[Any, Any]]:
        image = self.image_dataset[idx]
        degraded_image, depth_map = self.cond_dataset[idx]
        return image, (degraded_image, depth_map)
