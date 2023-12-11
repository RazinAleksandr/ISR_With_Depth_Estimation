import numpy as np


class PSNRMetric:
    def __init__(self):
        """Initialize the PSNRMetric class."""
        pass

    def __call__(self, true, predicted):
        """Compute psnr metrics when the class instance is called."""
        return self.compute_psnr_metrics(true, predicted)

    def compute_psnr_metrics(self, true, predicted):
        """Compute the PSNR value between two images."""
        mse = np.mean((true - predicted) ** 2)
        if mse == 0:
            # If the images are identical, the PSNR is infinite
            return float('inf')
        max_pixel = 255
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        metrics = {
            "PSNR": psnr
        }

        return metrics
    