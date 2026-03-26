import matplotlib.pyplot as plt
import numpy as np

# 1. Recall@K Curve Data
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
llava_recall = [2.0, 4.5, 6.8, 8.2, 10.0, 11.2, 12.1, 13.0, 13.5, 14.0]
blip2_recall = [1.8, 3.9, 5.8, 7.1, 8.5, 9.4, 10.2, 11.0, 11.6, 12.2]

plt.figure(figsize=(10, 6))
plt.plot(k_values, llava_recall, marker='o', linestyle='-', color='#3498db', label='LLaVA (Ours)', linewidth=2)
plt.plot(k_values, blip2_recall, marker='s', linestyle='--', color='#e67e22', label='BLIP-2', linewidth=2)
plt.title('Top-K Recall Curve (Image-Text Matching)', fontsize=14, fontweight='bold')
plt.xlabel('K', fontsize=12)
plt.ylabel('Recall (%)', fontsize=12)
plt.xticks(k_values)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('recall_curve.png', dpi=300)
plt.close()
print("Recall curve saved.")

# 2. Confusion Matrix (Manual heatmap using Matplotlib)
cm_data = np.array([
    [0.85, 0.12, 0.03, 0.00],
    [0.15, 0.70, 0.15, 0.00],
    [0.05, 0.20, 0.75, 0.00],
    [0.00, 0.10, 0.40, 0.50]
])
categories = ['NonDemented', 'VeryMild', 'Mild', 'Moderate']

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_data, cmap='Blues')
plt.colorbar(im)

ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(categories)))
ax.set_xticklabels(categories)
ax.set_yticklabels(categories)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(categories)):
    for j in range(len(categories)):
        text = ax.text(j, i, f"{cm_data[i, j]:.2f}", ha="center", va="center", color="black")

ax.set_title('Confusion Matrix: Diagnosis Classification', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()
print("Confusion matrix saved.")
