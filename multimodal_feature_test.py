import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

# Path settings
BASE_DIR = "well-documented-alzheimers-dataset/versions/2/"
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")

# Device setting
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def synthesize_clinical_text(row):
    """Convert metadata to clinical text for BioBERT."""
    text = (f"Patient {row['subject_id']}: A {row['age']}-year-old {row['gender']} patient. "
            f"Education: {row['education']} years. SES: {row['ses']}. "
            f"MMSE: {row['mmse']}, CDR: {row['cdr']}. "
            f"eTIV: {row['etiv']}, nWBV: {row['nwbv']}, ASF: {row['asf']}.")
    return text

class MultiModalFeatureExtractor:
    def __init__(self):
        print("Loading CLIP (Image Encoder)...")
        self.clip_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_name).to(device)
        
        print("Loading BioBERT (Text Encoder)...")
        self.biobert_name = "dmis-lab/biobert-v1.1"
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(self.biobert_name)
        self.biobert_model = AutoModel.from_pretrained(self.biobert_name).to(device)

    def extract_image_features(self, image_path):
        """Extract image features (512-dim) using CLIP."""
        full_path = os.path.join(BASE_DIR, image_path)
        image = Image.open(full_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def extract_text_features(self, clinical_text):
        """Extract text features (768-dim) using BioBERT."""
        inputs = self.biobert_tokenizer(clinical_text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = self.biobert_model(**inputs)
            text_features = outputs.last_hidden_state[:, 0, :]
        return text_features

def main():
    df = pd.read_csv(METADATA_PATH)
    sample_row = df.iloc[0]
    
    image_path = sample_row['sample_image_path']
    clinical_text = synthesize_clinical_text(sample_row)
    
    print(f"Subject ID: {sample_row['subject_id']}")
    print(f"Clinical Text: {clinical_text}")
    print(f"Image Path: {image_path}")
    
    extractor = MultiModalFeatureExtractor()
    
    img_feat = extractor.extract_image_features(image_path)
    txt_feat = extractor.extract_text_features(clinical_text)
    
    print(f"Image Feature Shape: {img_feat.shape}")
    print(f"Text Feature Shape: {txt_feat.shape}")
    
    fused_feat = torch.cat((img_feat, txt_feat), dim=1)
    print(f"Fused Feature Shape: {fused_feat.shape}")

if __name__ == "__main__":
    main()
