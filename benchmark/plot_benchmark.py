import json
import matplotlib.pyplot as plt
import numpy as np

with open('benchmark_comparison.json', 'r') as f:
    data = json.load(f)

plt.figure(figsize=(12, 6))

# Colors for different models
colors = ['#4F46E5', '#10B981', '#F59E0B']
models = ['TVM', 'Quantized', 'Base Model']

load_times = [data[model.lower()]['load_times']['mean'] for model in ['tvm', 'quantized', 'base']]
inference_times = [data[model.lower()]['inference_times']['mean'] for model in ['tvm', 'quantized', 'base']]

# Set width of each bar and positions of the bars
width = 0.35
x = np.arange(len(models))

# Create bars
bars1 = plt.bar(x - width/2, load_times, width, label='Load Time', color=colors)
bars2 = plt.bar(x + width/2, inference_times, width, label='Inference Time', color=[c + '80' for c in colors])

# Customize the plot
plt.title('Model Performance Comparison', fontsize=14, pad=20)
plt.ylabel('Time (milliseconds)')
plt.xticks(x, ["Sparse compiled model", "Quantized model", "Model"])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add value labels on the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom')

add_value_labels(bars1)
add_value_labels(bars2)


plt.tight_layout()

plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as benchmark_comparison.png")