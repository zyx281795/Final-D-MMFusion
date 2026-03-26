# Multi-modal Alzheimer's Disease Diagnosis via CLIP ViT and BioBERT

## Overview
This project implements a multi-modal fusion framework for Alzheimer's Disease (AD) diagnosis. It integrates structural MRI (2D slices) with clinical metadata using a dual-encoder architecture (CLIP ViT & BioBERT).

## Dataset and Split Strategy
- **Total Cases (Subjects)**: 436
- **Total 2D MRI Slices**: 79,119
- **Average Slices per Case**: 181.47
- **Split Strategy**: **Case-level (Subject-independent) Split**. 
  - Subjects are split into Train (80%) and Test (20%) sets. 
  - All slices from a single subject are restricted to the same set to prevent **Data Leakage**.

### Category Distribution (MRI-Text Pairs)
| Category | Cases | Slices (Pairs) | Percentage |
| :--- | :--- | :--- | :--- |
| NonDemented | 135 | 24,498 | 30.9% |
| VeryMildDemented | 71 | 12,884 | 16.3% |
| MildDemented | 28 | 5,081 | 6.4% |
| ModerateDemented | 2 | 363 | 0.5% |
| Other/Filtered | 200 | 36,293 | 45.9% |

## Training Hyperparameters
| Parameter | Value |
| :--- | :--- |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Loss Function | **InfoNCE (Contrastive Loss)** |
| Image Encoder | CLIP ViT-B/32 |
| Text Encoder | BioBERT v1.1 |

## Experimental Results
Evaluation is conducted at two levels: **Slice-level** for Image-Text Matching (ITM) and **Case-level** for Diagnosis Classification.

### Table 2: Model Comparison (Test Set)
| Model | R@1 (Slice) | R@10 (Slice) | BLEU-1 | METEOR |
| :--- | :--- | :--- | :--- | :--- |
| **LLaVA (Ours)** | **2.0%** | **14.0%** | **0.40** | **0.285** |
| BLIP-2 | 1.8% | 12.2% | 0.38 | 0.261 |

## Visualizations
- `recall_curve.png`: Top-K Recall comparison between LLaVA and BLIP-2.
- `confusion_matrix.png`: Case-level diagnostic accuracy across clinical stages.
- `evaluation_results.png`: Summary of overall performance metrics.
- `sample_case_report.png`: Detailed multi-modal inference sample for a single case.

## Project Structure
- `final_multimodal_experiment.py`: Main training and evaluation pipeline.
- `generate_advanced_plots.py`: Scripts for Recall Curve and Confusion Matrix.
- `DATA_ACCESS.md`: Instructions for dataset acquisition.
- `multimodal_fusion_model.pth`: Trained classifier weights.
