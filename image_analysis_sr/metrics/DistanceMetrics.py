import numpy as np
import matplotlib.pyplot as plt


class DistanceMetrics:
    def __init__(self):
        """Initialize the DistanceMetrics class."""
        pass

    def __call__(self, true, predicted):
        """Compute L1 and L2 distance metrics when the class instance is called."""
        return self.compute_distances(true, predicted)

    def compute_distances(self, true, predicted):
        """Compute the L1 and L2 distances between two images."""
        # Calculate L1 and L2 distances
        true = true.astype('float32')
        predicted = predicted.astype('float32')

        l1_distance = np.abs(true - predicted)
        l2_distance = np.sqrt((true - predicted) ** 2)

        # Normalize distances for display
        l1_norm = self.normalize(l1_distance)
        l2_norm = self.normalize(l2_distance)

        # Create matplotlib figures
        metrics = {
            "L1": self.create_figure(l1_norm),
            "L2": self.create_figure(l2_norm)
        }

        return metrics

    def normalize(self, distance):
        """Normalize the distance values to a 0-1 scale."""
        distance_min = np.min(distance)
        distance_max = np.max(distance)
        return (distance - distance_min) / (distance_max - distance_min)

    def create_figure(self, distance):
        """Create a matplotlib figure for the distance."""
        fig, ax = plt.subplots()
        cax = ax.imshow(distance, cmap='hot', interpolation='nearest')
        plt.colorbar(cax)
        return fig