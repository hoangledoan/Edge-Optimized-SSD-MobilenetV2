# Compression Module

## Overview
This module provides various model compression techniques to optimize the MobileNetV2-SSD object detection model for deployment on edge devices. It includes methods for weight pruning, quantization, and fine-tuning to regain the accuracy.

## Features
- **Unstructural Pruning**: Magnitude pruning, remove redudant weights based on their magnitude
- **K-Means Quantization**: Clusters similar weight values to decrease bit-width usage.

## Usage
### Pruning
Run the following command to prune the model:
```bash
python fine_grain_prune.py --model_path models_parameters/back_model.pth --dataset_path dataset/back/ --target_accuracy 80 --finetune
```
To save time, you can remove `--finetune` and set `--target_accuracy` for higher accuracy (85 for example) 
### Quantization
To quantize the model weights:
```bash
python compression/k_mean.py
```



