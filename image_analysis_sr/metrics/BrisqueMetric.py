from brisque import BRISQUE


class BrisqueMetric:
    def __init__(self):
        """Initialize the BRISQUE class."""
        self.obj = BRISQUE(url=False)

    def __call__(self, image):
        """Compute brisque metrics when the class instance is called."""
        return self.compute_quality_metrics(image)
    
    def compute_quality_metrics(self, img):
        """Compute the average brightness of the given image."""
        img_copy = img.copy()
        quality = self.obj.score(img_copy)
        
        metrics = {
            "Quality BRISQUE": quality
        }

        return metrics