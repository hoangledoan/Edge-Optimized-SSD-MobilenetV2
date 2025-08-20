import random
import argparse
import numpy as np
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite
from model_profile import *
from evaluate import *
from utils.misc import store_labels
from run_training import test, train
from utils.misc import Timer
from object_detection.ssd import MatchPrior
from datasets_processing.open_images import OpenImagesDataset
from utils import config as cfg
from training.data_preprocessing import TrainAugmentation, TestTransform
from utils import config as cfg
import json
# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class FineGrainedPruner:
    """
    Implements fine-grained weight pruning for neural networks.
    
    This class provides methods to prune model weights based on their magnitude,
    allowing for different sparsity levels across different layers.
    """
    
    def __init__(self, model, sparsity_dict):
        """
        Initialize the pruner with a model and sparsity dictionary.
        
        Args:
            model: The neural network model to be pruned
            sparsity_dict: Dictionary mapping layer names to desired sparsity levels
        """
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @staticmethod
    @torch.no_grad()
    def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
        """
        Prune individual weight tensors based on magnitude.
        
        Example:
            >>> # Create a sample 2x3 tensor
            >>> tensor = torch.tensor([[1.0, -2.0, 0.1],
            ...                       [0.2, -0.5, 3.0]])
            >>> sparsity = 0.5  # Remove 50% of weights
            >>> mask = fine_grained_prune(tensor, sparsity)
            >>> print(mask)
            tensor([[1, 1, 0],
                    [0, 0, 1]])
            >>> # The pruned tensor will keep only the larger magnitude values:
            >>> print(tensor)
            tensor([[1.0, -2.0, 0.0],
                    [0.0, 0.0, 3.0]])
        
        Args:
            tensor: Weight tensor to be pruned
            sparsity: Target sparsity level (0.0 to 1.0)
            
        Returns:
            torch.Tensor: Binary mask where 1s indicate weights to keep
        """
        sparsity = min(max(0.0, sparsity), 1.0)
        if sparsity == 1.0:
            tensor.zero_()
            return torch.zeros_like(tensor)
        elif sparsity == 0.0:
            return torch.ones_like(tensor)
            
        num_elements = tensor.numel()
        num_zeros = round(num_elements * sparsity)
        importance = torch.abs(tensor)
        threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]
        mask = importance > threshold
        tensor.mul_(mask)
        return mask
    
    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        """
        Apply pruning across the entire model.
        
        Example:
            >>> # Define sparsity levels for different layers
            >>> sparsity_dict = {
            ...     'conv1.weight': 0.3,  # Prune 30% of conv1 weights
            ...     'conv2.weight': 0.5,  # Prune 50% of conv2 weights
            ...     'fc.weight': 0.7      # Prune 70% of fc weights
            ... }
            >>> pruner = FineGrainedPruner(model, sparsity_dict)
            >>> masks = pruner.prune(model, sparsity_dict)
            >>> # masks will contain binary tensors for each layer
            >>> print(masks['conv1.weight'].shape)  # Shape matches layer
            torch.Size([64, 3, 3, 3])
        
        Args:
            model: Neural network model to prune
            sparsity_dict: Dictionary mapping layer names to sparsity levels
            
        Returns:
            dict: Pruning masks for each layer
        """
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1 and name in sparsity_dict:  # only prune conv and fc weights in the dict
                masks[name] = FineGrainedPruner.fine_grained_prune(param, sparsity_dict[name])
        return masks
    
    @torch.no_grad()
    def apply(self, model):
        """
        Apply stored pruning masks to a model.
        
        Args:
            model: Model to apply masks to
            
        Returns:
            model: Pruned model
        """
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]
        return model

class SensitivityScan:
    """
    Analyzes model sensitivity to weight pruning across different layers.
    
    This class implements a systematic approach to test how different levels
    of pruning affect model accuracy, helping identify which layers can be
    pruned more aggressively.
    
    Example:
        >>> # Create a sensitivity scanner
        >>> scanner = SensitivityScan(
        ...     scan_step=0.1,    # Test sparsity in steps of 10%
        ...     scan_start=0.1,   # Start at 10% sparsity
        ...     scan_end=0.8      # Test up to 80% sparsity
        ... )
        >>> # Sample results structure:
        >>> sparsities, accuracies = scanner.run_scanning(model)
        >>> print(sparsities)  # Sparsity levels tested
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        >>> # accuracies[0] shows impact on first layer:
        >>> print(accuracies[0])  # Accuracy at each sparsity for layer 1
        [95.5, 94.8, 94.2, 93.5, 92.8, 91.2, 89.5]
    """
    
    def __init__(self, scan_step: float, scan_start: float, scan_end: float):
        """
        Initialize sensitivity scanning parameters.
        
        Args:
            scan_step: Increment between sparsity levels
            scan_start: Starting sparsity level
            scan_end: Maximum sparsity level to test
        """
        self.scan_step = scan_step
        self.scan_start = scan_start
        self.scan_end = scan_end
        self.accuracy_threshold = args.target_accuracy
        self.start_layer = args.start_layer
        self.best_model_state = None
        self.best_sparsity_config = {}
        
        # Training configurations imported from run_training.py
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.fine_tune_epochs = args.fine_tune_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    def run_scanning(self, model):
        """
        Run sensitivity analysis on the model.
        
        For each layer after start_layer, gradually increases sparsity level
        until accuracy drops below threshold. Then reverts to the last good
        sparsity, fine-tunes the model, and moves to the next layer.
        
        Args:
            model: The neural network model to analyze
            
        Returns:
            tuple: (sparsity levels tested, accuracy measurements for each layer)
        """
        sparsities = np.arange(start=self.scan_start, stop=self.scan_end + 0.001, step=self.scan_step)
        layer_sparsities = []
        
        named_conv_weights = [(name, param) for (name, param) in model.base_net.named_parameters() if param.dim() > 1]
        
        
        # Create data loaders for fine-tuning if needed
        train_loader, val_loader = self._prepare_dataloaders(args.dataset_path)
        
        for i_layer, (name, param) in enumerate(named_conv_weights):
            if i_layer >= self.start_layer:
                print(f"\nProcessing Layer {name}: ")
                
                # Clone original parameter for reset if needed
                original_param = param.data.clone()
                
                last_good_sparsity = 0.0
                last_good_accuracy = 100.0  # Start with assumption of perfect accuracy
                last_good_param = original_param.clone()
                
                # Try gradually increasing sparsity levels
                for sparsity in sparsities:
                    # Apply pruning
                    FineGrainedPruner.fine_grained_prune(param, sparsity=sparsity)
                    acc = evaluate_accuracy(model, args.dataset_path)
                    acc = acc * 100  # Convert to percentage  
                    print(f'sparsity={sparsity:.2f}: accuracy={acc:.2f}%')
                    
                    # If accuracy drops below threshold, break and use the last good sparsity
                    if acc < self.accuracy_threshold:
                        print(f"Accuracy dropped below threshold {self.accuracy_threshold}% at sparsity {sparsity:.2f}")
                        break
                    
                    
                    last_good_sparsity = sparsity
                    last_good_accuracy = acc
                    last_good_param = param.data.clone()
                
                # Restore the last good pruning level
                param.data.copy_(last_good_param)
                
                # Store this layer's best sparsity
                self.best_sparsity_config[name] = last_good_sparsity
                layer_sparsities.append(last_good_sparsity)
                
                print(f'\nSelected sparsity for {name}: {last_good_sparsity:.2f} (accuracy: {last_good_accuracy:.2f}%)\n')
                
                # Fine-tune only if we found a non-zero sparsity level
                if last_good_sparsity > 0:
                    print(f"Fine-tuning with frozen pruned layers...")
                    
                    # Freeze all previously pruned layers
                    self._freeze_layers_before(model)
                    
                    # Fine-tune the model
                    final_acc = self._fine_tune(model, train_loader, val_loader)
                    
                    print(f"After fine-tuning: accuracy={final_acc:.2f}%")
                    
                    # Unfreeze all layers for next iteration
                    self._unfreeze_all_layers(model)
        
        # Save the final model state
        self.best_model_state = {name: param.data.clone() for name, param in model.named_parameters()}
        
        return sparsities, layer_sparsities

    def run_scanning_no_finetune(self, model):
        """
        Apply optimal pruning by finding maximum sparsity for each layer.
        
        For each layer after start_layer, increases sparsity until accuracy drops
        below the threshold, then applies the last working sparsity before moving
        to the next layer.
        
        Args:
            model: The neural network model to prune
            
        Returns:
            dict: Map of layer names to their applied sparsity levels
        """
        sparsities = np.arange(start=self.scan_start, stop=self.scan_end+0.0001, step=self.scan_step)
        named_conv_weights = [(name, param) for (name, param) in model.base_net.named_parameters() if param.dim() > 1]
        layer_sparsity_map = {}
        
        # Initial accuracy 
        initial_accuracy = evaluate_accuracy(model, args.dataset_path) * 100
        print(f"Initial accuracy: {initial_accuracy:.2f}%")
        
        for i_layer, (name, param) in enumerate(named_conv_weights):
            if i_layer > self.start_layer:
                print(f"\nPruning Layer {name}: ")
                
                # Store original parameter values for this layer
                original_param = param.data.clone()
                last_good_param = original_param.clone()
                last_good_sparsity = 0.0
                last_good_accuracy = initial_accuracy
                
                for sparsity in sparsities:
                    # Apply pruning to this layer
                    FineGrainedPruner.fine_grained_prune(param, sparsity=sparsity)
                    acc = evaluate_accuracy(model, args.dataset_path) * 100
                    print(f'sparsity={sparsity:.2f}: accuracy={acc:.2f}%')
                    
                    if acc >= self.accuracy_threshold:
                        # This sparsity is acceptable, save it
                        last_good_param = param.data.clone()
                        last_good_sparsity = sparsity
                        last_good_accuracy = acc
                    else:
                        # Accuracy dropped too much, stop and use last good sparsity
                        print(f'Accuracy dropped below threshold ({acc:.2f}% < {self.accuracy_threshold:.2f}%)')
                        print(f'Reverting to last good sparsity: {last_good_sparsity:.2f}')
                        break
                
                # Apply the last good sparsity level before moving to next layer
                if last_good_sparsity > 0:
                    param.data.copy_(last_good_param)
                    layer_sparsity_map[name] = last_good_sparsity
                    print(f'Applied sparsity {last_good_sparsity:.2f} to {name} (accuracy: {last_good_accuracy:.2f}%)')
                else:
                    # No acceptable sparsity found, keep original weights
                    param.data.copy_(original_param)
                    print(f'No viable sparsity found for {name}, keeping original weights')
        
        
        return layer_sparsity_map

    def _prepare_dataloaders(self, path):
        """
        Prepare data loaders for training and validation.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            tuple: train_loader, val_loader
        """

        config = cfg
        train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
        target_transform = MatchPrior(config.priors, config.center_variance,
                                      config.size_variance, 0.4)
        test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
        
        datasets = []
        dataset = OpenImagesDataset(path,
                transform=train_transform, target_transform=target_transform,
                dataset_type="train")
        label_file = os.path.join(args.checkpoint_folder, "labels.txt")
        store_labels(label_file, dataset.class_names)
        logging.info(dataset)
        num_classes = len(dataset.class_names)

        datasets.append(dataset)
        logging.info(f"Stored labels into file {label_file}.")
        train_dataset = ConcatDataset(datasets)
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, args.batch_size,
                                shuffle=True)

        logging.info("Prepare Validation datasets.")
        val_dataset = OpenImagesDataset(path,
                                            transform=test_transform, target_transform=target_transform,
                                            dataset_type="valid")
        logging.info(val_dataset)
        logging.info("validation dataset size: {}".format(len(val_dataset)))

        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _freeze_layers_before(self, model):
        """
        Freeze only the layers that have already been pruned.
        
        Args:
            model: The model to freeze layers in
            current_layer_idx: Index of the current layer
            named_weights: List of (name, param) tuples for weight tensors
        """
        with torch.no_grad():
            # Only freeze layers that have a sparsity level set (have been pruned)
            for name in self.best_sparsity_config:
                # Get the base name of the layer (without .weight suffix)
                base_name = name.rsplit('.', 1)[0]
                # Find and freeze the corresponding layer
                for module_name, module in model.base_net.named_modules():
                    if module_name == base_name:
                        print(f"Freezing pruned layer: {module_name}")
                        for param in module.parameters():
                            param.requires_grad = False
    
    def _unfreeze_all_layers(self, model):
        """
        Unfreeze all layers in the model.
        
        Args:
            model: The model to unfreeze layers in
        """
        for param in model.parameters():
            param.requires_grad = True
    
    def _fine_tune(self, model, train_loader, val_loader):
        """
        Fine-tune the model after pruning.
        
        Args:
            model: The pruned model to fine-tune
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            float: Accuracy after fine-tuning
        """
        from utils.misc import Timer
        from utils.multibox_loss import MultiboxLoss
        from utils import config as cfg
        
        config = cfg
        device = self.device
        
        # Create criterion
        criterion = MultiboxLoss(config.priors, iou_threshold=0.4, neg_pos_ratio=3,
                               center_variance=0.1, size_variance=0.2, device=device)
        
        # Setup optimizer - only optimize unfrozen layers
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.learning_rate, 
                                  momentum=self.momentum, 
                                  weight_decay=self.weight_decay)
        
        # Setup scheduler
        scheduler = MultiStepLR(optimizer, milestones=[80,100],
                              gamma=0.1)
        
        model.to(device)
        train(train_loader, val_loader, model, criterion, optimizer,
             device=device, debug_steps=50, batch_size= self.batch_size,
            epochs=self.fine_tune_epochs, scheduler= scheduler, model_name="models_parameters/temp_model.pth")
        
        # Evaluate the fine-tuned model
        acc = evaluate_accuracy(model, args.dataset_path)
        return acc * 100 
        
    def save_model(self, model, path):
        """
        Save the best pruned model.
        
        Args:
            model: The model to save
            path: Path to save the model to
        """
        if self.best_model_state is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self.best_model_state:
                        param.data.copy_(self.best_model_state[name])
            model.save(path)
            
            # Also save the sparsity configuration for reference
            with open(f"{path}_sparsity_config.txt", 'w') as f:
                for name, sparsity in self.best_sparsity_config.items():
                    f.write(f"{name}: {sparsity:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate accuracy of object detection model with fine-grained pruning and fine-tuning.")
    parser.add_argument("--model_path", type=str, help="Path to the model weights", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path to the test dataset", required=True)
    parser.add_argument("--target_accuracy", type=float, help="Target to prune up to this accuracy", required=True)
    parser.add_argument("--start_layer", type=int, 
                        help="Start pruning from this layer. Total number of layers is 52, the starting layers are important, but the latter layers can be easily pruned",
                        default=15, required=False)    
    
    # Added configs from run_training.py
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--fine_tune_epochs', default=5, type=int, help='Number of epochs for fine-tuning')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Use CUDA to train model')
    parser.add_argument('--checkpoint_folder', default='models_parameters/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--finetune', action= "store_true", help='Finetune for more accuracy')
    args = parser.parse_args()
    model = create_mobilenetv2_ssd_lite(2, is_test=True)
    model.load(args.model_path, load_quantized= False)
    model.eval()
    sensitivity_scan = SensitivityScan(scan_step=0.1, scan_start=0.1, scan_end=1.0)
    if args.finetune:
        print("Pruning with finetuning")
        sparsities, accuracies = sensitivity_scan.run_scanning(model=model)
    else:
        print("Prunning with no finetuning")
        sparsities, accuracies = sensitivity_scan.run_scanning_no_finetune(model=model)

    with open("sparsity.json", 'w') as f:
        json.dump(sensitivity_scan.best_sparsity_config, f, indent=4)

    pruned_model_path = f"sparse_model_{args.target_accuracy}%.pth"
    sensitivity_scan.save_model(model, pruned_model_path)
    print(f"Saved pruned model to {pruned_model_path}")