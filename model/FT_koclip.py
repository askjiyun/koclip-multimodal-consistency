"""
Fine-tuned koCLIP Training & Evaluation Script

This script fine-tunes a Korean CLIP model (Bingsu/clip-vit-large-patch14-ko) on an
Instagram dataset of image–hashtag pairs to classify five levels of semantic
consistency (0–4).

Pipeline
--------
1) Data loading & split
   - Reads CSV with columns: ["Image Filename", "Hashtag", "Score"].
   - Normalizes labels from 1–5 to 0–4.
   - Performs an 80/20 stratified train/validation split.

2) Dataset & preprocessing
   - Loads images from the given folder and resizes to 224×224.
   - Uses AutoProcessor to tokenize text and prepare pixel values (max_length=77).

3) Model
   - Base: AutoModel (Bingsu/clip-vit-large-patch14-ko) to obtain text/image embeddings.
   - Head: A classifier over concatenated embeddings (image_embeds ⊕ text_embeds)
     → Linear(1536→256) → ReLU → Dropout(0.3) → Linear(256→5).

4) Training
   - Loss: CrossEntropy with class weights (computed from label distribution).
   - Optimizer: Adam (lr=5e-5, weight_decay=1e-4).
   - Scheduler: StepLR(step_size=5, gamma=0.1).
   - Mixed precision: torch.amp (GradScaler + autocast) with gradient accumulation.

5) Evaluation & logging
   - Reports per-epoch: train/val loss, val accuracy, macro/weighted F1, and a
     scikit-learn classification report.
   - Saves the fine-tuned weights as `fine_tuned_koclip.pth`.
   - Plots and saves `training_results.png` (loss, accuracy, F1 curves, final summary).

Usage
-----
    python FT_koclip.py

Notes
-----
- Requires a CSV and an image directory accessible via paths set in the script.
- If a GPU is available, CUDA will be used automatically.
- Adjust batch size, epochs, learning rate, and accumulation steps to fit memory.
"""


import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt


# Step 1. Define the dataset
class KoreanClipDataset(Dataset):
    def __init__(self, dataframe, image_folder, processor):
        self.data = dataframe
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = f"{self.image_folder}/{row['Image Filename']}"
        text = row['Hashtag']
        label = row['Score']  # Ground-truth label

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and resize the image
        image = Image.open(image_path).convert("RGB").resize((224, 224))

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",  # Pad to the maximum length
            truncation=True,       # Truncate overly long text
            max_length=77          # CLIP's maximum text length
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# Step 2. Define the model (add a classifier on top of the base model)
class FineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes=5, freeze_base=False):
        super(FineTunedCLIP, self).__init__()
        self.base_model = base_model

        # Set text and image embedding dimensions
        text_embedding_dim = self.base_model.text_projection.out_features
        image_embedding_dim = self.base_model.visual_projection.out_features

        # Define the classifier head
        self.classifier = nn.Sequential(
            nn.Linear(text_embedding_dim + image_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Optionally freeze the CLIP backbone
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        # Forward pass through the base model
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract image and text embeddings
        image_embeddings = outputs.image_embeds  # (batch_size, 768)
        text_embeddings = outputs.text_embeds   # (batch_size, 768)

        # Concatenate image and text embeddings
        combined_embeddings = torch.cat((image_embeddings, text_embeddings), dim=1)  # (batch_size, 1536)

        # Predict consistency score with the classifier
        logits = self.classifier(combined_embeddings)
        return logits



def main():
    # Load and preprocess data
    # TODO: Replace the paths below with your own data paths
    data_path = 'path/to/your/combined_data.csv'  # Path to the CSV file
    image_folder = 'path/to/your/image/folder'    # Path to the image directory

    df = pd.read_csv(data_path)
    
    # Normalize labels (1–5 -> 0–4)
    df['Score'] = df['Score'] - 1

    # Load base model and processor
    repo = "Bingsu/clip-vit-large-patch14-ko"
    base_model = AutoModel.from_pretrained(repo)
    processor = AutoProcessor.from_pretrained(repo)

    print(f"Model Architecture:\n{base_model}")

    # Train/Test Split (8:2)
    train_idx, test_idx = train_test_split(
        df.index, 
        test_size=0.2,
        stratify=df['Score'],
        random_state=42
    )

    train_dataset = KoreanClipDataset(df.loc[train_idx], image_folder, processor)
    test_dataset = KoreanClipDataset(df.loc[test_idx], image_folder, processor)

    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Check label distribution
    print("Train Dataset Label Distribution:")
    print(df.loc[train_idx, 'Score'].value_counts().sort_index())
    print("\nTest Dataset Label Distribution:")
    print(df.loc[test_idx, 'Score'].value_counts().sort_index())
    print(f"\nTest Set Size: {len(test_dataset)}")

    # Hyperparameters
    learning_rate = 5e-5
    epochs = 20
    weight_decay = 1e-4
    gradient_accumulation_steps = 2

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['Score']),
        y=df['Score']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = FineTunedCLIP(base_model, num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    scaler = GradScaler()
    best_val_loss = float('inf')

    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_macro_f1s = []
    val_weighted_f1s = []

    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
            
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = criterion(outputs, labels) / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps

        scheduler.step()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
            
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
               )
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                    
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Compute and store per-epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_accuracy = correct / total
        epoch_macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        epoch_weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        val_macro_f1s.append(epoch_macro_f1)
        val_weighted_f1s.append(epoch_weighted_f1)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        print(f"Validation Accuracy: {epoch_accuracy:.4f}")
        print(f"Macro F1 Score: {epoch_macro_f1:.4f}")
        print(f"Weighted F1 Score: {epoch_weighted_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))

    # Save the trained model
    torch.save(model.state_dict(), "fine_tuned_koclip.pth")
    print("모델이 저장되었습니다: fine_tuned_koclip.pth")

    # Visualize training results
    plt.figure(figsize=(15, 10))

    # 1. Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs+1), val_accuracies, 'g-')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # 3. F1 Scores
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs+1), val_macro_f1s, 'c-', label='Macro F1')
    plt.plot(range(1, epochs+1), val_weighted_f1s, 'm-', label='Weighted F1')
    plt.title('F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # 4. Final performance summary
    plt.subplot(2, 2, 4)
    final_metrics = {
        'Train Loss': train_losses[-1],
        'Val Loss': val_losses[-1],
        'Accuracy': val_accuracies[-1],
        'Macro F1': val_macro_f1s[-1],
        'Weighted F1': val_weighted_f1s[-1]
    }
    y_pos = np.arange(len(final_metrics))
    plt.barh(y_pos, list(final_metrics.values()))
    plt.yticks(y_pos, list(final_metrics.keys()))
    plt.title('Final Performance Metrics')
    plt.xlabel('Value')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print final metrics
    print("\n=== Final Performance Metrics ===")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Macro F1 Score: {val_macro_f1s[-1]:.4f}")
    print(f"Weighted F1 Score: {val_weighted_f1s[-1]:.4f}")

    # Print best epoch metrics
    best_epoch = np.argmin(val_losses) + 1
    print("\n=== Best Performance ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Loss: {min(val_losses):.4f}")
    print(f"Corresponding Accuracy: {val_accuracies[best_epoch-1]:.4f}")
    print(f"Corresponding Macro F1: {val_macro_f1s[best_epoch-1]:.4f}")
    print(f"Corresponding Weighted F1: {val_weighted_f1s[best_epoch-1]:.4f}")


if __name__ == "__main__":
    main()
