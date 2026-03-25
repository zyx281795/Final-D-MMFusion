import matplotlib.pyplot as plt
from PIL import Image
import os

# Paths
BASE_DIR = "well-documented-alzheimers-dataset/versions/2/"
image_path = os.path.join(BASE_DIR, "VeryMildDemented/VeryMildDemented/OAS1_0003_MR1_1.nii_slice_137.png")

# Sample Data
subject_id = "OAS1_0003_MR1"
age, gender, mmse, cdr = 73, "Female", 27.0, 0.5
category, nwbv = "VeryMildDemented", 0.708

# Text & ITM Results
ref_report = "Moderate hippocampus atrophy consistent with early AD."
gen_report = "73y patient shows early hippocampus changes, MMSE 27."
bleu_score = 0.4286
itm_results = [("VeryMild (Match)", 0.8752), ("MildDemented", 0.7410), ("NonDemented", 0.5298)]

# Create Figure
fig = plt.figure(figsize=(14, 9), facecolor='white')

# 1. MRI Image Plot (Left)
ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3)
img = Image.open(image_path).convert("RGB")
ax_img.imshow(img)
ax_img.set_title("Input MRI Slice (OAS1_0003_MR1)", fontsize=14, fontweight='bold', pad=10)
ax_img.axis('off')

# 2. Subject Profile (Top Right)
ax_prof = plt.subplot2grid((3, 3), (0, 1), colspan=2)
ax_prof.axis('off')
prof_text = (f"Age: {age} | Gender: {gender}\n"
             f"MMSE Score: {mmse}/30.0 | CDR Grade: {cdr}\n"
             f"Diagnosis: {category} | nWBV Index: {nwbv}")
ax_prof.text(0.05, 0.4, prof_text, fontsize=13, family='monospace', linespacing=2)
ax_prof.set_title("Clinical Metadata Profile", fontsize=15, fontweight='bold', loc='left', pad=15)

# 3. Text Generation Assessment (Middle Right)
ax_gen = plt.subplot2grid((3, 3), (1, 1), colspan=2)
ax_gen.axis('off')
gen_info = (f"REF: {ref_report}\n"
            f"GEN: {gen_report}\n"
            f"BLEU-1 Score: {bleu_score:.4f}")
ax_gen.text(0.05, 0.4, gen_info, fontsize=12, style='italic', color='#2c3e50', linespacing=1.8)
ax_gen.set_title("Medical Report Generation Analysis", fontsize=15, fontweight='bold', loc='left', pad=15)

# 4. ITM Retrieval Ranking (Bottom Right)
ax_itm = plt.subplot2grid((3, 3), (2, 1), colspan=2)
labels, scores = [r[0] for r in itm_results], [r[1] for r in itm_results]
colors = ['#27ae60', '#f39c12', '#c0392b']
bars = ax_itm.barh(labels, scores, color=colors, alpha=0.85)
ax_itm.invert_yaxis()
ax_itm.set_xlim(0, 1.0)
ax_itm.set_xlabel("Similarity Confidence Score", fontsize=11)
ax_itm.set_title("ITM: Cross-modal Retrieval Top-3", fontsize=15, fontweight='bold', loc='left', pad=15)

for bar in bars:
    ax_itm.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.4f}', va='center', fontweight='bold')

plt.tight_layout(pad=4.0)
plt.savefig('sample_case_report.png', dpi=300)
print("Case report image saved: sample_case_report.png")
