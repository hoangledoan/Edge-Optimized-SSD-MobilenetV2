import torch
import cv2
import time
import numpy as np
from typing import List, Tuple
import psutil
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite
from compression.k_mean import QuantizedModel

class BatchQuantizedPredictor:
    def __init__(self, quantized_model, image_size=300, mean=[127, 127, 127], std=[128.0, 128.0, 128.0]):
        self.quantized_model = quantized_model
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
    
    def predict_batch(self, images: List[np.ndarray], top_k: int = 10, prob_threshold: float = 0.6) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_tensor = torch.stack([self.preprocess(img) for img in images])
        
        with torch.no_grad():
            for name, param in self.quantized_model.base_net.named_parameters():
                if name in self.quantized_model.base_net.compressed_state_dict:
                    compressed_tensor = self.quantized_model.base_net.compressed_state_dict[name]
                    param.data = compressed_tensor.decompress()
                elif name in self.quantized_model.base_net.non_quantized_params:
                    param.data = self.quantized_model.base_net.non_quantized_params[name].clone()
            
            confidence, locations = self.quantized_model(batch_tensor)
            
            # Compress back
            for name, param in self.quantized_model.base_net.named_parameters():
                if name in self.quantized_model.base_net.compressed_state_dict:
                    param.data = torch.zeros(1)
            
        return confidence, locations

def get_process_memory():
    return psutil.Process().memory_info().rss

def run_quantized_batch_comparison(
    model_path: str,
    image_path: str,
    batch_sizes: List[int],
    num_warmup: int = 5,
    num_runs: int = 5
):
    model = create_mobilenetv2_ssd_lite(2, is_test=True, device="cpu")
    model.load(model_path)
    model.eval()
    
    quantized_model = QuantizedModel(model.base_net)
    quantized_model.load(QUANTIZED_SAVE_PATH)
    model.base_net = quantized_model
    batch_predictor = BatchQuantizedPredictor(model)
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    print("\n" + "="*60)
    print("Quantized Batch Processing Benchmark (CPU)")
    print("="*60)
    
    print("\nWarm-up Phase:")
    for _ in range(num_warmup):
        batch_predictor.predict_batch([original_image])
    
    single_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        batch_predictor.predict_batch([original_image])
        single_times.append(time.time() - start_time)

    
    single_mean = np.mean(single_times)
    single_std = np.std(single_times)
    
    print(f"Baseline: {single_mean:.4f} ± {single_std:.4f}s")
    
    print("\nBatch Results:")
    print("Size | Total(s) | Per Image(s) | Speedup")
    print("-"*60)
    
    results = []
    for batch_size in batch_sizes:
        batch_images = [original_image] * batch_size
        batch_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            confidence, locations = batch_predictor.predict_batch(batch_images)
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
        
        batch_mean = np.mean(batch_times)
        per_image = batch_mean / batch_size
        speedup = single_mean / per_image
        
        print(f"{batch_size:^4d} | {batch_mean:^8.4f} | {per_image:^11.4f} | "
              f"{speedup:^7.2f}x ")

if __name__ == "__main__":
    MODEL_PATH = "models_parameters/finetuned_sparse_model.pth"
    IMAGE_PATH = "image.jpg"
    QUANTIZED_SAVE_PATH = "compression/quantized_model.pth"
    BATCH_SIZES = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    
    run_quantized_batch_comparison(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        batch_sizes=BATCH_SIZES,
        num_warmup=5,
        num_runs=5
    )