from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import time
import torch
import numpy as np
import os
from typing import List, Tuple, Dict

class BatchPredictor:
    def __init__(self, net, image_size, mean, std):
        self.net = net
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image"""
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
    
    def predict_batch(self, images: List[np.ndarray], top_k: int = 10, prob_threshold: float = 0.6) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Predict on a batch of images"""
        # Stack the images and process
        batch_tensor = torch.stack([self.preprocess(img) for img in images])
        device = next(self.net.parameters()).device
        batch_tensor = batch_tensor.to(device)
        with torch.no_grad():
            confidence, locations = self.net(batch_tensor)
        return confidence, locations 
        
def run_batch_comparison(
    image_path: str,
    batch_sizes: List[int],
    num_warmup: int = 5,
    num_runs: int = 5
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label_path = os.path.join(current_dir, "models_parameters", "labels.txt")
    model_path = os.path.join(current_dir, "models_parameters", "finetuned_sparse_model.pth")
    net = create_mobilenetv2_ssd_lite(2, is_test=True)
    net.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    standard_predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    batch_predictor = BatchPredictor(net, 300, [127, 127, 127], [128.0, 128.0, 128.0])
    original_image = cv2.imread(image_path)
    
    print("\n" + "="*50)
    print("Batch Processing Benchmark")
    
    # Perform warm-up iterations
    print("\nWarm-up Phase:")
    print("-"*50)
    warmup_times = []
    for i in range(num_warmup):
        start_time = time.time()
        boxes, labels, probs = standard_predictor.predict(original_image, 10, 0.6)
        inference_time = time.time() - start_time
        warmup_times.append(inference_time)    
    
    # Single image baseline
    print("\nSingle Image Baseline:")
    print("-"*50)
    single_times = []
    for i in range(num_runs):
        start_time = time.time()
        boxes, labels, probs = standard_predictor.predict(original_image, 10, 0.6)
        inference_time = time.time() - start_time
        single_times.append(inference_time)    
    single_mean = np.mean(single_times)
    single_std = np.std(single_times)
    print(f"\nBaseline Statistics:")
    print(f"Average: {single_mean:.4f} ± {single_std:.4f} seconds")
    print("-"*50)
    
    # Batch processing
    print("\nBatch Processing Results:")
    print("-"*50)
    print("Batch Size | Total Time (s) | Per Image (s) | Speedup")
    print("-"*50)
    
    for batch_size in batch_sizes:
        batch_images = [original_image] * batch_size
        batch_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            confidence, locations  = batch_predictor.predict_batch(batch_images, 10, 0.6)
            inference_time = time.time() - start_time
            batch_times.append(inference_time)
        
        batch_mean = np.mean(batch_times)
        per_image_mean = batch_mean / batch_size
        speedup = single_mean / per_image_mean
        
        print(f"{batch_size:^10d} | {batch_mean:^13.4f} | {per_image_mean:^13.4f} | {speedup:^7.2f}x")
    
    print("-"*50)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "models_parameters", "image.jpg")
    batch_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    run_batch_comparison(
        image_path=image_path,
        batch_sizes=batch_sizes,
        num_warmup=5,
        num_runs=5
    )