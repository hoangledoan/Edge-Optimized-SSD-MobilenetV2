import torch
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from training.data_preprocessing import TestTransform
from utils.multibox_loss import MultiboxLoss
from utils import config
from datasets_processing.open_images import OpenImagesDataset  
from object_detection.ssd import MatchPrior
from torch.utils.data import DataLoader
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
def calculate_metrics(net, test_loader, device, num_classes):
    """
    Calculate accuracy, precision, recall, and F1 score metrics
    """
    net.eval()
    metrics = {
        'correct_predictions': 0,
        'total_predictions': 0,
        'true_positives': np.zeros(num_classes),
        'false_positives': np.zeros(num_classes),
        'false_negatives': np.zeros(num_classes),
        'class_predictions': [],
        'class_targets': [],
        'confidence_scores': []
    }
    
    with torch.no_grad():
        for images, boxes, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            confidence, locations = net(images)
            
            predicted_classes = torch.argmax(confidence, dim=2)
            confidence_scores, _ = torch.max(confidence, dim=2)
            
            # Mask for positive (object) priors
            pos_mask = labels > 0
            
            # Store predictions and targets for overall metrics
            metrics['class_predictions'].extend(predicted_classes[pos_mask].cpu().numpy())
            metrics['class_targets'].extend(labels[pos_mask].cpu().numpy())
            metrics['confidence_scores'].extend(confidence_scores[pos_mask].cpu().numpy())
            
            # Update counts
            correct_batch = (predicted_classes[pos_mask] == labels[pos_mask]).sum().item()
            total_batch = pos_mask.sum().item()
            
            metrics['correct_predictions'] += correct_batch
            metrics['total_predictions'] += total_batch
            
            # Update per-class metrics
            for c in range(num_classes):
                class_mask = labels == c
                pred_mask = predicted_classes == c
                
                metrics['true_positives'][c] += torch.logical_and(class_mask, pred_mask).sum().item()
                metrics['false_positives'][c] += torch.logical_and(torch.logical_not(class_mask), pred_mask).sum().item()
                metrics['false_negatives'][c] += torch.logical_and(class_mask, torch.logical_not(pred_mask)).sum().item()

    # Calculate final metrics
    results = {}
    
    # Overall accuracy
    results['accuracy'] = metrics['correct_predictions'] / metrics['total_predictions'] if metrics['total_predictions'] > 0 else 0
    
    # Overall metrics using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        metrics['class_targets'],
        metrics['class_predictions'],
        average='weighted'
    )
    
    results['overall_metrics'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return results

def print_metrics(results, class_names):
    """Print accuracy, precision, recall, and F1 score metrics"""
    print("\n=== Model Accuracy Metrics ===")
    print("\nOverall Metrics:")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['overall_metrics']['precision']*100:.2f}%")
    print(f"Recall: {results['overall_metrics']['recall']*100:.2f}%")
    print(f"F1 Score: {results['overall_metrics']['f1']*100:.2f}%")
    

if __name__ == "__main__":   
    label_path = "/home/hoangledoan/smartlib/computer_vision/models_parameters/labels.txt"
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    net.load("/home/hoangledoan/smartlib/computer_vision/models_parameters/front_model.pth")
    
    # Prepare test dataset
    dataset_path = "/home/hoangledoan/student_counting"
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_dataset = OpenImagesDataset(dataset_path, transform=test_transform, 
                                     dataset_type="test", target_transform=target_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    results = calculate_metrics(net, test_loader, device, len(class_names))
    with open('model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print_metrics(results, class_names)
    