
import matplotlib.pyplot as plt
import numpy as np

# Data from previous evaluation
itm_metrics = ['Recall@1', 'Recall@5', 'Recall@10']
itm_values = [2.00, 10.00, 14.00] # Percentage

text_metrics = ['BLEU-1', 'METEOR']
text_values = [40.00, 28.50] # 0.4000 -> 40.0, 0.2850 -> 28.5

# Combine for plotting
metrics = itm_metrics + text_metrics
values = itm_values + text_values
colors = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c'] # Blue for ITM, Red for Text

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metrics, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Styling
ax.set_ylim(0, 50)
ax.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Multi-modal Evaluation Results: ITM & Text Generation', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#3498db', lw=4, label='Image-Text Matching (Recall@K)'),
    Line2D([0], [0], color='#e74c3c', lw=4, label='Text Generation Quality')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300)
print("Plot successfully saved as 'evaluation_results.png'")
