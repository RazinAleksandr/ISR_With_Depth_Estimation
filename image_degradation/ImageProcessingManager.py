import os
from PIL import Image
from tqdm import tqdm

from AdvancedImageDegradation import AdvancedImageDegradation


class ImageProcessingManager:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.advanced_degradation = AdvancedImageDegradation()

    def analyse(
            self,
            images_path,
            apply_bsr,
            ):
        for img_filename in tqdm(os.listdir(images_path)):
            img_path = os.path.join(images_path, img_filename)
            modified_image = self._modify_single(
                img_path,
                apply_bsr
                )
            modified_image.save(os.path.join(self.log_dir, img_filename))

    def _modify_single(
            self, 
            img_path,
            apply_bsr
            ):
        
        image = Image.open(img_path)

        if apply_bsr:
            degradated = self.advanced_degradation.apply_bsr_degradation(image)

        return degradated