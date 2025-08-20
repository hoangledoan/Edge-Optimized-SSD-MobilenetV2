import copy
import math
import random
from collections import namedtuple
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from fast_pytorch_kmeans import KMeans
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from compression.evaluate import *
import cv2
from compression.tensor_utils import CompressedTensor, QuantizedModel


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class KMean:
    def __init__(self, model):
        self.model = model
        self.base_net = model.base_net
        self.quantized_model = None
        self.compressed_params = {}

    @torch.no_grad()
    def k_means_quantize(self, fp32_tensor: torch.Tensor):
        """
        Compress a single tensor using K-means clustering.
        """
        n_clusters = 2**8 

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1))
        labels = labels.to(torch.int16)
        centroids = kmeans.centroids.squeeze()

        return CompressedTensor(
            labels=labels,
            centroids=centroids,
            original_shape=fp32_tensor.shape
        )
    
    def quantize_model(self):
        """
        Compress the weight tensor and leave out the bias.

        Returns:
        Model with compressed weights
        """
        self.quantized_model = QuantizedModel(self.base_net)

        for name, param in self.base_net.named_parameters():
            # Checking if the tensor is the weights and not the bias.
            if param.dim() > 1:
                compressed_tensor = self.k_means_quantize(param.data)
                self.quantized_model.compressed_state_dict[name] = compressed_tensor
                # Save to fine-tuning
                self.compressed_params[name] = compressed_tensor
            # If it is the bias, we don't quantize it.
            else:  
                self.quantized_model.non_quantized_params[name] = param.data.clone()
        return self.quantized_model
    
    @torch.no_grad()
    def update_centroid(self, update_centroids=True):
        """
        Update centroids of the quantized model based on the current parameters.
        This is used for fine-tuning quantized models.
        
        Args:
            update_centroids: Whether to update centroids or not
        
        Returns:
            Dictionary of updated compressed parameters
        """
        if not update_centroids or not self.compressed_params:
            return self.compressed_params
            
        updated_params = {}
        
        for name, param in self.base_net.named_parameters():
            if name not in self.compressed_params:
                continue
                
            compressed_tensor = self.compressed_params[name]
            labels = compressed_tensor.labels
            centroids = compressed_tensor.centroids
            original_shape = compressed_tensor.original_shape
            
            # Get the current parameter values
            current_values = param.data.view(-1)
            
            # Create a new tensor to store updated centroids
            new_centroids = torch.zeros_like(centroids)
            
            # Update each centroid with the mean of all values assigned to it
            for k in range(centroids.numel()):
                # If no values are assigned to this centroid, keep the old value
                if (labels == k).sum() == 0:
                    new_centroids[k] = centroids[k]
                else:
                    # Simple one-line update: average of all values assigned to this centroid
                    new_centroids[k] = torch.mean(current_values[labels == k])
            
            # Create a new compressed tensor with updated centroids
            updated_tensor = CompressedTensor(
                labels=labels,
                centroids=new_centroids,
                original_shape=original_shape
            )
            
            # Store the updated tensor
            updated_params[name] = updated_tensor
            
            # Update the model's parameters with the decompressed values
            param.data.copy_(updated_tensor.decompress())
        
        # Update the compressed parameters
        self.compressed_params = updated_params
        
        # Also update the quantized model if it exists
        if self.quantized_model:
            self.quantized_model.compressed_state_dict = updated_params
            
        return updated_params





if __name__ == "__main__":
    ORIGINAL_MODEL_PATH = "models_parameters/back_model.pth"
    QUANTIZED_SAVE_PATH = "models_parameters/quantized.pth"
    TEST_IMAGE_PATH = "image.jpg"

    model = create_mobilenetv2_ssd_lite(2, is_test=True)
    model.load(ORIGINAL_MODEL_PATH)
    model.eval()

    quantizer = KMean(model)
    quantized_model = quantizer.quantize_model()
    
    # Save batch norm stats before changing base_net
    bn_stats = {}
    for name, module in model.base_net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_stats[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
                'weight': module.weight.clone() if module.weight is not None else None,
                'bias': module.bias.clone() if module.bias is not None else None
            }
    
    # Create save dictionary
    save_dict = {
        'base_net': {
            'compressed_state_dict': {
                name: tensor.save_to_dict() 
                for name, tensor in quantized_model.compressed_state_dict.items()
            },
            'non_quantized_params': quantized_model.non_quantized_params,
            'batch_norm': bn_stats
        },
        'source_layer_add_ons': model.source_layer_add_ons.state_dict(),
        'extras': model.extras.state_dict(),
        'classification_headers': model.classification_headers.state_dict(),
        'regression_headers': model.regression_headers.state_dict(),
    }
    
    # Save the model
    torch.save(save_dict, QUANTIZED_SAVE_PATH)
    # print(f"Quantized model saved to {QUANTIZED_SAVE_PATH}")
    
    # # Verify loading
    # loaded_model = create_mobilenetv2_ssd_lite(2, is_test=True)
    # loaded_model.load(QUANTIZED_SAVE_PATH, load_quantized=True)
    # loaded_model.eval()

    # predictor = create_mobilenetv2_ssd_lite_predictor(loaded_model, candidate_size=200)
    # img = cv2.imread(TEST_IMAGE_PATH)
    # if img is None:
    #     raise ValueError(f"Could not read image at {TEST_IMAGE_PATH}")

    # boxes, labels, probs = predictor.predict(img, top_k=10, prob_threshold=0.5)
    # print(boxes)
    # img_copy1 = img.copy()
    # for i in range(boxes.size(0)):
    #     box = boxes[i, :]
    #     label = f"Class {labels[i]}"
    #     prob = probs[i]
    #     print(f"{label}: {prob:.2f} at location {box}")
        
    #     x1, y1, x2, y2 = [int(coord) for coord in box]
    #     cv2.rectangle(img_copy1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     text = f"{label}: {prob:.2f}"
    #     cv2.putText(img_copy1, text, (x1, y1-10), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # output_path = "detected_output1.jpg"
    # cv2.imwrite(output_path, img_copy1)