import torch
from torch import nn
from object_detection.mobilenetv2 import MobileNetV2
    
class CompressedTensor:
    """
    A compressed tensor as a new representation for memory saving.
    
    Instead of storing weights as 32-bit numbers, we:
    1. Find common weight values (centroids) using K-means clustering and store them in 8 bits.
    2. Store the orginal weights as a list of indices of 8 bits, points to the corresponding centroid.

    The representation can be in lower representation, as 4 or 2 bits. However pytorch only support
    the minimum int8.
    
    Example:
    Original weights: [1.2, 2.72, 5.831, 0.103, 5.202] in 32 bits
    After compression:
    - Centroids: [2.0, 5.0] (stored as float32)
    - Labels: [0, 0, 1, 0, 1] (stored as uint16, where 0 points to 2.0 and 1 points to 5.0)

    In run time, we load the centroids and labels. Decompress it to the original weights and do inference.
    This approach saves the memory, however it introduces computation overhead as the labels needs to 
    multiply with its corresponding centroids before it do inference.
    """
    def __init__(self, labels, centroids, original_shape):
        """
        Initialize a compressed tensor.
        
        Args:
            labels (torch.Tensor): Indices pointing to centroids
            centroids (torch.Tensor): Unique weight values (normalized)
            original_shape (tuple): Shape of the original tensor
            tensor_min (float): Minimum value of original tensor
            tensor_max (float): Maximum value of original tensor
        """
        self.labels = labels
        self.centroids = centroids
        self.original_shape = original_shape

    def save_to_dict(self):
        """
        Save as a dictionary for easy loading.
        """
        return {
            'labels': self.labels,
            'centroids': self.centroids,
            'original_shape': self.original_shape,
        }
    
    @classmethod
    def load_from_dict(cls, state_dict):
        """
        Create a CompressedTensor from a saved dictionary.
        """
        return cls(
            labels=state_dict['labels'],
            centroids=state_dict['centroids'],
            original_shape=state_dict['original_shape'],
        )
    
    def decompress(self):
        """
        Convert compressed format back to original tensor shape and value range.
        """
        reconstructed = self.centroids[self.labels.to(torch.long)].reshape(self.original_shape)
        return reconstructed

class QuantizedModel(nn.Module):
    """
    A wrapper specifically designed for MobileNetV2 features that handles compressed weights.
    """
    def __init__(self, original_model, width_mult=1.0):
        """
        Initialize a quantized model wrapper.
        Args:
            original_model: The MobileNetV2 features to compress
            width_mult: Width multiplier for MobileNetV2
        """
        super().__init__()
        self.compressed_state_dict = {}  # Stores compressed weights
        self.non_quantized_params = {}  # Stores params that can't be compressed (e.g. biases)
        self.original_model = original_model  # Keep reference to original features
        self.width_mult = width_mult
        
        # Create a new features model that will hold decompressed weights
        self.decompressed_features = MobileNetV2(width_mult=self.width_mult).features
        
    def _decompress_params(self):
        """
        Decompresses all compressed parameters and updates the decompressed features.
        This should be called after loading compressed weights and before any inference.
        """
        with torch.no_grad():
            for name, param in self.decompressed_features.named_parameters():
                # print(name)
                if name in self.compressed_state_dict:
                    # Decompress weights using CompressedTensor's decompress method
                    decompressed = self.compressed_state_dict[name].decompress()
                    param.data.copy_(decompressed)
                elif name in self.non_quantized_params:
                    # Assign non-quantized parameters directly
                    param.data.copy_(self.non_quantized_params[name])
        
    def __getitem__(self, idx):
        """
        Access the decompressed features layers.
        Args:
            idx: Index or slice to access layers
        Returns:
            Selected layer(s) with decompressed weights
        """
        return self.decompressed_features[idx]
