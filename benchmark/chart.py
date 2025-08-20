import json
import matplotlib.pyplot as plt
import numpy as np

with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)

plot_data = {
    # 'Accuracy': metrics['accuracy'] * 100,
    'Precision': metrics['overall_metrics']['precision'] * 100,
    'Recall': metrics['overall_metrics']['recall'] * 100,
    'F1 Score': metrics['overall_metrics']['f1'] * 100
}

labels = list(plot_data.keys())
values = list(plot_data.values())

plt.figure(figsize=(10, 6))

bars = plt.bar(labels, values, color='#4F46E5', width=0.6)

plt.title('Model (Back) Performance Metrics', fontsize=14, pad=20)
plt.ylabel('Percentage (%)', fontsize=12)
plt.ylim(0, 105)  

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('front_model_performance.png', dpi=300, bbox_inches='tight')