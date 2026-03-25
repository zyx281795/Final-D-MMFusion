import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
import random

# Path settings
BASE_DIR = "well-documented-alzheimers-dataset/versions/2/"
METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiModalADModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MultiModalADModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, text_inputs):
        with torch.no_grad():
            img_feat = self.clip_model.get_image_features(**images)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_outputs = self.biobert_model(**text_inputs)
            txt_feat = text_outputs.last_hidden_state[:, 0, :]
        fused = torch.cat((img_feat, txt_feat), dim=1)
        return self.classifier(fused)

def synthesize_text(row):
    return (f"Patient {row['subject_id']}: Age {row['age']}, MMSE {row['mmse']}, "
            f"CDR {row['cdr']}, nWBV {row['nwbv']}.")

class ADDataset(Dataset):
    def __init__(self, df, clip_processor, biobert_tokenizer):
        self.df = df
        self.clip_processor = clip_processor
        self.biobert_tokenizer = biobert_tokenizer
        self.label_map = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(BASE_DIR, row['sample_image_path'])
        image = Image.open(img_path).convert("RGB")
        img_inputs = self.clip_processor(images=image, return_tensors="pt")
        text = synthesize_text(row)
        txt_inputs = self.biobert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
        label = self.label_map.get(row['category'], 0)
        return {
            "images": {k: v.squeeze(0) for k, v in img_inputs.items()},
            "text": {k: v.squeeze(0) for k, v in txt_inputs.items()},
            "label": torch.tensor(label)
        }

def run_experiment():
    print(f"Starting Multi-modal Fusion Experiment on {device}...")
    df = pd.read_csv(METADATA_PATH).dropna(subset=['category', 'sample_image_path'])
    
    indices = list(range(len(df)))
    random.shuffle(indices)
    subset_indices = indices[:20]
    split = int(0.8 * len(subset_indices))
    train_indices, test_indices = subset_indices[:split], subset_indices[split:]
    
    train_df, test_df = df.iloc[train_indices], df.iloc[test_indices]
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    bio_tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    
    train_loader = DataLoader(ADDataset(train_df, clip_proc, bio_tok), batch_size=4, shuffle=True)
    test_loader = DataLoader(ADDataset(test_df, clip_proc, bio_tok), batch_size=4)
    
    model = MultiModalADModel().to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training phase...")
    for epoch in range(2):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            imgs = {k: v.to(device) for k, v in batch['images'].items()}
            txts = {k: v.to(device) for k, v in batch['text'].items()}
            labels = batch['label'].to(device)
            outputs = model(imgs, txts)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1} complete.")

    # --- SAVE WEIGHTS ---
    weight_path = "multimodal_fusion_model.pth"
    # Only saving the classifier state to keep file size small (backbones are frozen)
    torch.save(model.classifier.state_dict(), weight_path)
    print(f"Successfully saved trained classifier weights to: {weight_path}")

    print("Evaluation phase...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            imgs = {k: v.to(device) for k, v in batch['images'].items()}
            txts = {k: v.to(device) for k, v in batch['text'].items()}
            labels = batch['label'].to(device)
            outputs = model(imgs, txts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0); correct += (predicted == labels).sum().item()
    
    print(f"Classification Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    run_experiment()
