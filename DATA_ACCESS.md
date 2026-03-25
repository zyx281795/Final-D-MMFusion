# Dataset Access and Structure

## Data Source
The data used in this project is derived from the **OASIS-1 (Open Access Series of Imaging Studies)** cross-sectional dataset. 

## Download Link
[Insert your cloud storage URL here (e.g., Google Drive, Dropbox, or Zenodo)]

## Dataset Structure
Once downloaded, ensure the data is organized in the following hierarchy:
```
well-documented-alzheimers-dataset/
└── versions/
    └── 2/
        ├── metadata.csv
        ├── NonDemented/
        ├── VeryMildDemented/
        ├── MildDemented/
        └── ModerateDemented/
```

## Preprocessing Details
1. **Image Conversion**: 3D NIfTI volumes were sliced into 2D PNG images (224x224 resolution).
2. **Normalization**: Images are normalized to RGB channels to maintain compatibility with CLIP pre-trained weights.
3. **Text Synthesis**: Metadata fields (Age, MMSE, CDR) are converted into structured natural language templates for BioBERT encoding.
