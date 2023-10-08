from torchvision.transforms import Resize, ToTensor


class BaseTransform:
    """
    A class for basic image data transformation.
    """
    def __init__(self, size=32, resize=True):
        self.size = size
        self.resize = resize

        if self.resize:
            self.resize_transform = Resize((self.size, self.size))
        
        self.to_tensor_transform = ToTensor()

    def __call__(self, sample):
        """
        Transforms the sample to PyTorch tensors.

        Args:
            sample (PIL.Image): The sample to transform.

        Returns:
            tensor (torch.Tensor): Transformed sample as a PyTorch tensor.
        """
        # Resize if required
        if self.resize:
            sample = self.resize_transform(sample)
        
        # Convert sample to tensor
        tensor_sample = self.to_tensor_transform(sample)

        return tensor_sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

