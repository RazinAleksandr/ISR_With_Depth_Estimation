from piq import total_variation, TVLoss
import torch


class TVIndexMetric:
    def __init__(self):
        """Initialize the TVIndexMetric class."""
        pass

    def __call__(self, image):
        """Compute tv metrics when the class instance is called."""
        return self.compute_tvindex_metrics(image)
    
    def compute_tvindex_metrics(self, img):
        """Compute the tv_index and tv_variance of the given image."""
        img_tensor = torch.tensor(img).permute(2, 0, 1)[None, ...] / 255.  # Convert to tensor and scale

        # Move to GPU if available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Calculate Total Variation
        tv_index = total_variation(img_tensor)
        tv_loss = TVLoss(reduction='none')(img_tensor)
        
        metrics = {
            "TV_index": tv_index.item(),
            "TV_loss": tv_loss.item(),
        }

        return metrics