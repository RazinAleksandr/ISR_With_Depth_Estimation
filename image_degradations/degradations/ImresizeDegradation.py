from PIL import Image


class ImresizeDownscale:
    """
    A class for image downscaling using imresize (bicubic interpolation, x2 downscale).
    """
    def __init__(self, scale_factor=0.5, interpolation=Image.BICUBIC):
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    def __call__(self, image):
        return self.compute_imresize_downscale(image)

    def compute_imresize_downscale(self, pil_image):
        original_size = pil_image.size
        downscaled_size = (int(original_size[0] * self.scale_factor), int(original_size[1] * self.scale_factor))
        
        # Downscale the image
        downscaled_img = pil_image.resize(downscaled_size, self.interpolation)

        return downscaled_img

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={}, interpolation={})'.format(self.scale_factor, self.interpolation)
