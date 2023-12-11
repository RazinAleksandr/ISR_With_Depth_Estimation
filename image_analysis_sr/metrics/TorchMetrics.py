import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, UniversalImageQualityIndex, TotalVariation, RelativeAverageSpectralError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
torch.manual_seed(42)


class TorchMetrics:
    def __init__(self):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.uqi = UniversalImageQualityIndex()
        self.tv = TotalVariation()
        self.rase = RelativeAverageSpectralError()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    def __call__(self, true, predicted):
        return self.compute_torch_metric(true, predicted)
    
    def compute_torch_metric(self, true, predicted):
        true_tensor = torch.tensor(true.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        predicted_tensor = torch.tensor(predicted.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)

        ssim = self.ssim(predicted_tensor, true_tensor).item()
        uqi = self.uqi(predicted_tensor, true_tensor).item()
        tv = self.tv(predicted_tensor).item()
        rase = self.rase(predicted_tensor, true_tensor).item()
        lpips = self.lpips(predicted_tensor * 2 - 1, true_tensor * 2 - 1).item()


        metrics = {
            "SSIM": ssim,
            "Universal Image Quality Index": uqi,
            "Total Variation": tv,
            "Relative Average Spectral Error": rase,
            "Learned Perceptual Image Patch Similarity": lpips
        }

        return metrics