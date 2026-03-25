# Multi-modal Alzheimer's Disease Diagnosis via CLIP ViT and BioBERT

## Overview
This project implements a multi-modal fusion framework for the diagnosis and classification of Alzheimer's Disease (AD). By integrating structural MRI imaging with clinical metadata, the model aligns visual features with medical linguistic context to improve diagnostic accuracy across various stages of dementia.

## Methodology
The framework utilizes a dual-encoder architecture:
- **Visual Encoder**: CLIP ViT-B/32 is employed to extract high-dimensional semantic features from 2D MRI slices.
- **Textual Encoder**: BioBERT (v1.1) is used to encode clinical metadata, including MMSE scores, CDR grades, and demographic information, into a medical linguistic embedding space.
- **Fusion & Classification**: A late-fusion strategy (concatenation) integrates the 512-dim visual features and 768-dim textual features, followed by a Multi-Layer Perceptron (MLP) for clinical stage prediction.

## Dataset Statistics
- Total Subjects: 436
- Total 2D MRI Slices: 79,119
- Average Slices per Case: 181.47

## Evaluation Metrics
The model is evaluated using two primary tasks:
1. **Image-Text Matching (ITM)**: Measured by Recall@K (R@1, R@5, R@10) to assess cross-modal retrieval performance.
2. **Medical Report Generation**: Evaluated via BLEU-1 to measure the alignment between model-generated summaries and clinical ground truth.

## Experimental Results
| Metric | Result |
| :--- | :--- |
| Recall@1 | 2.00% |
| Recall@10 | 14.00% |
| BLEU-1 | 0.4000 |

## Project Structure
- `final_multimodal_experiment.py`: Main training and evaluation pipeline.
- `generate_plot.py`: Script for overall performance visualization.
- `generate_sample_report.py`: Individual case study visualization.
- `DATA_ACCESS.md`: Instructions for dataset acquisition and structure.
- `requirements.txt`: Environment dependencies.
