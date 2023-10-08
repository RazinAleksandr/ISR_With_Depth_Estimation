from PIL import ImageFilter


class BSRDegradation:
    """
    A class for BSR image degradation.
    """
    def __init__(self, radius=1):
        self.radius = radius
    
    def __call__(self, pil_image):
        """
        Applies BSR degradation to the image.

        Args:
            pil_image (PIL.Image): The image to degrade.

        Returns:
            PIL.Image: The degraded image.
        """
        # Apply Gaussian blur
        degraded_img = pil_image.filter(ImageFilter.GaussianBlur(self.radius))
        
        return degraded_img

    def __repr__(self):
        return self.__class__.__name__ + '(radius={})'.format(self.radius)
