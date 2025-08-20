import tvm
import torch
import numpy as np
import cv2
import time
import os
from statistics import mean, stdev
from training.data_preprocessing import PredictionTransform
from object_detection.mobilenet_v2_ssd import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import json

class ModelBenchmark:
    def __init__(self, compiled_model_path):
        self.compiled_model_path = compiled_model_path
        # Load compiled model
        self.lib = tvm.runtime.load_module(compiled_model_path)
        self.dev = tvm.cpu(0)
        self.module = tvm.contrib.graph_executor.GraphModule(self.lib["default"](self.dev))
        
    def measure_load_time(self):
        start_time = time.perf_counter()
        lib = tvm.runtime.load_module(self.compiled_model_path)
        dev = tvm.cpu(0)
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        end_time = time.perf_counter()
        return end_time - start_time

    def measure_inference_time(self, input_data):
        start_time = time.perf_counter()
        self.module.set_input("input0", tvm.nd.array(input_data))
        self.module.run()
        scores = torch.from_numpy(self.module.get_output(0).numpy())[0]
        boxes = torch.from_numpy(self.module.get_output(1).numpy())[0]
        end_time = time.perf_counter()
        return end_time - start_time, (scores, boxes)

class BaseModel:
    def __init__(self, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes
        self.net = create_mobilenetv2_ssd_lite(num_classes, is_test=True, device="cpu")
        self.predictor = None

    def measure_load_time(self):
        start_time = time.perf_counter()
        self.net.load(self.model_path, load_quantized=False)
        self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200)
        end_time = time.perf_counter()
        return end_time - start_time

    def measure_inference_time(self, img):
        start_time = time.perf_counter()
        boxes, labels, probs = self.predictor.predict(img, 10, 0.3)
        end_time = time.perf_counter()
        return end_time - start_time, (boxes, labels, probs)

class QuantizedModel:
    def __init__(self, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes
        self.net = create_mobilenetv2_ssd_lite(num_classes, is_test=True, device="cpu")
        self.predictor = None

    def measure_load_time(self):
        start_time = time.perf_counter()
        self.net.load(self.model_path, load_quantized=True)
        self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200)
        end_time = time.perf_counter()
        return end_time - start_time

    def measure_inference_time(self, img):
        start_time = time.perf_counter()
        boxes, labels, probs = self.predictor.predict(img, 10, 0.3)
        end_time = time.perf_counter()
        return end_time - start_time, (boxes, labels, probs)

def run_comparison_benchmark(tvm_model_path, pytorch_model_path, quantized_model_path, input_image_path, num_classes, num_runs=10, num_warmup=5):
    # Prepare input data once
    transform = PredictionTransform(
        size=300,
        mean=np.array([127, 127, 127]),
        std=128.0
    )
    img = cv2.imread(input_image_path)
    input_data = transform(img).unsqueeze(0).numpy()
    
    # Initialize benchmarks
    tvm_benchmark = ModelBenchmark(tvm_model_path)
    base_benchmark = BaseModel(pytorch_model_path, num_classes)
    quantized_benchmark = QuantizedModel(quantized_model_path, num_classes)
    
    results = {
        'tvm': {
            'load_times': [],
            'inference_times': [],
            'total_times': []
        },
        'pytorch': {
            'load_times': [],
            'inference_times': [],
            'total_times': []
        },
        'quantized': {
            'load_times': [],
            'inference_times': [],
            'total_times': []
        }
    }

    # Warmup phase with TVM
    print(f"\nRunning {num_warmup} warm-up iterations...")
    for i in range(num_warmup):
        print(f"Warm-up iteration {i+1}/{num_warmup}")
        _, _ = tvm_benchmark.measure_inference_time(input_data)

    # Benchmark phase
    print(f"\nRunning {num_runs} benchmark iterations...")
    for i in range(num_runs):
        print(f"Iteration {i+1}/{num_runs}")
        
        # TVM benchmarking
        tvm_load_time = tvm_benchmark.measure_load_time()
        tvm_inference_time, _ = tvm_benchmark.measure_inference_time(input_data)
        
        results['tvm']['load_times'].append(tvm_load_time)
        results['tvm']['inference_times'].append(tvm_inference_time)
        results['tvm']['total_times'].append(tvm_load_time + tvm_inference_time)

        # PyTorch base model benchmarking
        base_load_time = base_benchmark.measure_load_time()
        base_inference_time, _ = base_benchmark.measure_inference_time(img)
        
        results['pytorch']['load_times'].append(base_load_time)
        results['pytorch']['inference_times'].append(base_inference_time)
        results['pytorch']['total_times'].append(base_load_time + base_inference_time)

        # PyTorch quantized model benchmarking
        quantized_load_time = quantized_benchmark.measure_load_time()
        quantized_inference_time, _ = quantized_benchmark.measure_inference_time(img)
        
        results['quantized']['load_times'].append(quantized_load_time)
        results['quantized']['inference_times'].append(quantized_inference_time)
        results['quantized']['total_times'].append(quantized_load_time + quantized_inference_time)

    # Calculate statistics
    stats = {'tvm': {}, 'pytorch': {}, 'quantized': {}}
    for framework in ['tvm', 'pytorch', 'quantized']:
        for key, values in results[framework].items():
            stats[framework][key] = {
                'mean': mean(values) * 1000,  # Convert to milliseconds
                'std': stdev(values) * 1000 if len(values) > 1 else 0,
                'min': min(values) * 1000,
                'max': max(values) * 1000
            }
    
    return stats

def print_comparison_results(stats):
    print("\nBenchmark Comparison Results (all times in milliseconds):")
    print("-" * 80)
    print(f"{'Operation':<15} {'Model':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)
    
    operations = ['load_times', 'inference_times', 'total_times']
    for operation in operations:
        operation_name = operation.replace('_', ' ').title()
        for framework in ['tvm', 'pytorch', 'quantized']:
            metrics = stats[framework][operation]
            framework_name = 'TVM' if framework == 'tvm' else 'PyTorch' if framework == 'pytorch' else 'Quantized'
            print(f"{operation_name:<15} {framework_name:<12} {metrics['mean']:>10.2f} "
                  f"{metrics['std']:>10.2f} {metrics['min']:>10.2f} {metrics['max']:>10.2f}")
        print("-" * 80)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    tvm_model_path = os.path.join(current_dir, "tvm_model/compiled_model.so")
    pytorch_model_path = os.path.join(current_dir, "models_parameters/model.pth")
    quantized_model_path = os.path.join(current_dir, "models_parameters/quantized1.pth")
    input_image_path = os.path.join(current_dir, "image.jpg")
    label_path = os.path.join(current_dir, "models_parameters/labels.txt")
    
    # Load class names
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)
    
    # Run benchmark comparison
    stats = run_comparison_benchmark(
        tvm_model_path,
        pytorch_model_path,
        quantized_model_path,
        input_image_path,
        num_classes,
        num_runs=10,
        num_warmup=5
    )
    
    print_comparison_results(stats)
    
    with open('benchmark_comparison.json', 'w') as f:
        json.dump(stats, f, indent=4)