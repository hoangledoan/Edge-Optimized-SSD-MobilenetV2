from torch.utils.data import DataLoader
import argparse
import random
import numpy as np

from object_detection.ssd import MatchPrior
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite
from datasets_processing.open_images import OpenImagesDataset
from utils import config
from training.data_preprocessing import TestTransform
from compression.model_profile import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def calculate_classification_accuracy(net, test_loader, device):
    """
    Calculate classification accuracy for object detection model. If the model fails to detect, 
    it is marked as wrong classified.
    
    Returns:
        Overall classification accuracy
    """
    net.eval()
    correct_predictions = 0
    total_predictions = 0


    with torch.no_grad():
        for images, boxes, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)

            confidence, locations = net(images)

            # Get predicted class for each prior box
            predicted_classes = torch.argmax(confidence, dim=2)
            
            # Mask for positive (object) priors
            pos_mask = labels > 0

            # Classification accuracy, including those who are not detected
            correct_batch = (predicted_classes[pos_mask] == labels[pos_mask]).sum().item()
            total_batch = pos_mask.sum().item()
            
            correct_predictions += correct_batch
            total_predictions += total_batch        
    
    classification_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return classification_accuracy

def evaluate_accuracy(model, dataset_path):
    """
    Evaluates the performance of a given model on a dataset for classification.
    """
    # dataset_path = "/home/hoangledoan/dataset/people_counting/"
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_dataset = OpenImagesDataset(dataset_path, transform=test_transform, 
                                     dataset_type="test", target_transform= target_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return calculate_classification_accuracy(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Evaluate accuracy of object detection model.")
    parser.add_argument("--model_path", type=str, help="Path to the model weights", required= True)
    parser.add_argument("--dataset_path", type=str, help="Path to the test dataset", required= True)
    args = parser.parse_args()
    model = create_mobilenetv2_ssd_lite(2, is_test= True)
    model.load(args.model_path)
    model.eval()
    accuracy = evaluate_accuracy(model, args.dataset_path)
    print(accuracy)
