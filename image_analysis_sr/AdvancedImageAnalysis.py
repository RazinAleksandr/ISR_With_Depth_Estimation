import numpy as np

from metrics import PSNRMetric, BrisqueMetric, DistanceMetrics


class AdvancedImageAnalysis:
    def __init__(self):
        pass

    def psnr_metric(self, true, predicted):
        """Calculate the blur level of a block or image."""
        psnr_calculation = PSNRMetric()
        return psnr_calculation(np.array(true), np.array(predicted))
    
    def brisque_metric(self, im):
        """Calculate the quality of a block or image using a specific metric like BRISQUE."""
        brusque_calculation = BrisqueMetric()
        return brusque_calculation(np.array(im))
    
    # def tv_metric(self, im):
    #     """Calculate the quality of a block or image using a specific metric like TV index."""
    #     tv_calculation = TVIndexMetric()
    #     return tv_calculation(np.array(im))

    def distance_metric(self, true, predicted):
        """Calculate the quality of a block or image using l1 l2 distances."""
        distance_calculation = DistanceMetrics()
        return distance_calculation(np.array(true), np.array(predicted))