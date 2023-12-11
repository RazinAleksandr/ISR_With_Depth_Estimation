from PIL import ImageFilter


class BSRDegradation:
    """
    A class for BSR image degradation.
    """
    def __init__(self, radius=3):
        self.radius = radius

    def __call__(self, image):
        return self.compute_bsr_degradation(image)

    def compute_bsr_degradation(self, pil_image):
        degraded_img = pil_image.filter(ImageFilter.GaussianBlur(self.radius))
        
        return degraded_img

    def __repr__(self):
        return self.__class__.__name__ + '(radius={})'.format(self.radius)