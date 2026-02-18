#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 2-2 KoClip Fine-tuning Model Evaluation Script

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


class KoCLIPDataset(Dataset):
    """Custom dataset class for Korean CLIP"""
    def __init__(self, dataframe, image_folder, processor):
        self.data = dataframe
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, row['Image Filename'])
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a fallback image (black image)
            image = Image.new('RGB', (224, 224), color='black')
            
        text = row["Hashtag"]
        label = int(row["Score"])

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class FrozenKoCLIPWithClassifier(nn.Module):
    """Korean CLIP with frozen encoder and trainable classifier head"""
    def __init__(self, model_name="Bingsu/clip-vit-large-patch14-ko", num_classes=5):
        super(FrozenKoCLIPWithClassifier, self).__init__()
        self.clip_model = AutoModel.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Trainable classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_model.config.projection_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.clip_model(
                pixel_values=pixel_values, 
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
        combined = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined)


def setup_environment():
    """Environment setup"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def load_and_prepare_data(data_path='/Jupyter/Insta/data/combined_data.csv'):
    """Load and prepare data"""
    try:
        df = pd.read_csv(data_path)
        df['Score'] = df['Score'] - 1  # Normalize to range 0–4
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Score distribution:\n{df['Score'].value_counts().sort_index()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def train_model(train_dataset, val_dataset, device, config, output_dir, model_name):
    """Train a single model"""
    print(f"\n[{model_name}] Training started")
    
    # Initialize model
    model = FrozenKoCLIPWithClassifier(num_classes=5).to(device)
    optimizer = optim.Adam(
        model.classifier.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Apply class weights
    criterion = nn.CrossEntropyLoss(weight=config['class_weights'].to(device))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Training logs
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"{model_name}, Epoch {epoch+1}/{config['epochs']}")
        
        for batch in train_pbar:
            try:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(pixel_values, input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total if total > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"{model_name}, Epoch {epoch + 1}/{config['epochs']}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")
        
        # Print classification report (only on final epoch)
        if epoch == config['epochs'] - 1:
            print("Final Classification Report:")
            print(classification_report(all_labels, all_predictions))

    # Save model
    model_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Full model saved: {model_path}")
    
    # Save classifier only
    classifier_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_classifier.pth")
    torch.save(model.classifier.state_dict(), classifier_path)
    print(f"Classifier saved: {classifier_path}")

    return model, all_labels, all_predictions, train_losses, val_losses, val_accuracies


def prepare_cross_validation(df, K=5):
    """Prepare cross-validation"""
    skf = StratifiedKFold(n_splits=K, random_state=42, shuffle=True)
    labels = df['Score'].values
    fold_indices = list(skf.split(df, labels))
    
    print(f"Prepared {K}-fold cross validation")
    for i, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    return fold_indices, labels


def train_single_fold(fold, train_dataset, val_dataset, device, config, output_dir):
    """Train a single fold"""
    model_name = f"Fold {fold + 1}/{config['K']}"
    model, fold_labels, fold_predictions, train_losses, val_losses, val_accuracies = train_model(
        train_dataset, val_dataset, device, config, output_dir, model_name
    )
    
    # Save fold-specific model
    fold_model_path = os.path.join(output_dir, f"frozen_koclip_fold_{fold + 1}.pth")
    torch.save(model.state_dict(), fold_model_path)
    print(f"Fold {fold + 1} model saved: {fold_model_path}")
    
    return model, fold_labels, fold_predictions, train_losses, val_losses, val_accuracies


def evaluate_and_save_results(all_labels, all_predictions, method_name, output_dir):
    """Evaluate and save results"""
    # Compute metrics
    mae = mean_absolute_error(all_labels, all_predictions)
    acc = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)
    report_dict = classification_report(all_labels, all_predictions, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"\n{method_name} Results:")
    print(f"MAE: {mae:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Filename prefix
    prefix = method_name.lower().replace(' ', '_').replace('-', '_')

    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {method_name}")
    plt.tight_layout()
    
    conf_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {conf_path}")

    # Save CSV file (quantitative metrics)
    metrics_df = pd.DataFrame({
        'Method': [method_name],
        'MAE': [mae],
        'Accuracy': [acc],
        'Macro_F1': [report_dict['macro avg']['f1-score']],
        'Weighted_F1': [report_dict['weighted avg']['f1-score']]
    })
    csv_path = os.path.join(output_dir, f"{prefix}_evaluation_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")

    # Save TXT file (full report)
    txt_path = os.path.join(output_dir, f"{prefix}_evaluation_metrics.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write(f"{method_name} Results\n\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Results saved: {txt_path}")

    # Save JSON file (classification report)
    json_path = os.path.join(output_dir, f"{prefix}_classification_report.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"Classification report saved: {json_path}")

    return mae, acc, report_dict


def plot_training_curves(train_losses, val_losses, val_accuracies, method_name, output_dir):
    """Visualize training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    if isinstance(train_losses[0], list):  # Cross-validation case
        for fold in range(len(train_losses)):
            axes[0].plot(train_losses[fold], label=f'Fold {fold+1} Train', alpha=0.7)
            axes[1].plot(val_losses[fold], label=f'Fold {fold+1} Val', alpha=0.7)
            axes[2].plot(val_accuracies[fold], label=f'Fold {fold+1} Val Acc', alpha=0.7)
    else:  # Single split case
        axes[0].plot(train_losses, label='Train Loss', alpha=0.7)
        axes[1].plot(val_losses, label='Val Loss', alpha=0.7)
        axes[2].plot(val_accuracies, label='Val Accuracy', alpha=0.7)
    
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Validation Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{method_name} - Training Curves')
    plt.tight_layout()
    
    prefix = method_name.lower().replace(' ', '_').replace('-', '_')
    curves_path = os.path.join(output_dir, f"{prefix}_training_curves.png")
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {curves_path}")


def data_80_20_split(df, dataset, config, device, output_dir):
    """Run training with 80:20 split"""
    print("\n" + "="*60)
    print("8:2 SPLIT TRAINING")
    print("="*60)
    
    # 80:20 split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Score']
    )
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Train model
    model, all_labels, all_predictions, train_losses, val_losses, val_accuracies = train_model(
        train_dataset, val_dataset, device, config, output_dir, "8:2 Split"
    )
    
    # Evaluate and save results
    mae, acc, report_dict = evaluate_and_save_results(
        all_labels, all_predictions, "Baseline 2-2 8:2 Split", output_dir
    )
    
    # Save training curves
    plot_training_curves(
        train_losses, val_losses, val_accuracies, "Baseline 2-2 8:2 Split", output_dir
    )
    
    return mae, acc, report_dict


def run_cross_validation(df, dataset, config, device, output_dir):
    """Run cross-validation"""
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Prepare cross-validation
    fold_indices, labels = prepare_cross_validation(df, config['K'])
    
    dataset_splits = [
        (Subset(dataset, train_idx), Subset(dataset, val_idx)) 
        for train_idx, val_idx in fold_indices
    ]
    
    # Execute cross-validation
    all_fold_labels = []
    all_fold_predictions = []
    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    
    for fold, (train_dataset, val_dataset) in enumerate(dataset_splits):
        model, fold_labels, fold_predictions, train_losses, val_losses, val_accuracies = train_single_fold(
            fold, train_dataset, val_dataset, device, config, output_dir
        )
        
        all_fold_labels.append(fold_labels)
        all_fold_predictions.append(fold_predictions)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_accuracies.append(val_accuracies)
        
        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Aggregate all fold results
    all_labels = []
    all_predictions = []
    
    for labels, predictions in zip(all_fold_labels, all_fold_predictions):
        all_labels.extend(labels)
        all_predictions.extend(predictions)
    
    # Final evaluation and save results
    mae, acc, report_dict = evaluate_and_save_results(
        all_labels, all_predictions, "Baseline 2-2 Cross-Validation", output_dir
    )
    
    plot_training_curves(
        all_train_losses, all_val_losses, all_val_accuracies, 
        "Baseline 2-2 Cross-Validation", output_dir
    )
    
    return mae, acc, report_dict


def main():
    print("=" * 60)
    print("Baseline 2-2 Korean CLIP Fine-tuning Evaluation")
    print("Both 8:2 Split and K-Fold Cross-Validation")
    print("=" * 60)
    
    try:
        # 1. Environment setup
        device = setup_environment()
        output_dir = "./results/"
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Configuration
        config = {
            'K': 5,  # Number of folds
            'learning_rate': 5e-5,
            'batch_size': 32,
            'epochs': 10,
            'weight_decay': 1e-4
        }
        
        # 3. Load data
        # TODO: Replace with your own data path
        data_path = 'path/to/your/combined_data.csv'  # CSV file path
        image_folder = 'path/to/your/image/folder'    # Image file path
        
        df = load_and_prepare_data(data_path)
        
        # 4. Compute class weights
        labels = df['Score'].values
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(labels), 
            y=labels
        )
        config['class_weights'] = torch.tensor(class_weights, dtype=torch.float)
        print(f"Class weights: {class_weights}")
        
        # 5. Prepare dataset and processor
        print("Loading Korean CLIP processor...")
        processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
        dataset = KoCLIPDataset(df, image_folder, processor)
        print("Dataset created successfully")
        
        # 6. Run 80:20 split
        mae_split, acc_split, report_split = data_80_20_split(
            df, dataset, config, device, output_dir
        )
        
        # 7. Run cross-validation
        mae_cv, acc_cv, report_cv = run_cross_validation(
            df, dataset, config, device, output_dir
        )
        
        # 8. Final comparison
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"8:2 Split Results:")
        print(f"  MAE: {mae_split:.4f}")
        print(f"  Accuracy: {acc_split:.4f}")
        print(f"  Macro F1: {report_split['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1: {report_split['weighted avg']['f1-score']:.4f}")
        
        print(f"\nCross-Validation Results:")
        print(f"  MAE: {mae_cv:.4f}")
        print(f"  Accuracy: {acc_cv:.4f}")
        print(f"  Macro F1: {report_cv['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1: {report_cv['weighted avg']['f1-score']:.4f}")
        
        # 9. Save comparison results
        comparison_df = pd.DataFrame({
            'Method': ['8:2 Split', 'Cross-Validation'],
            'MAE': [mae_split, mae_cv],
            'Accuracy': [acc_split, acc_cv],
            'Macro_F1': [report_split['macro avg']['f1-score'], report_cv['macro avg']['f1-score']],
            'Weighted_F1': [report_split['weighted avg']['f1-score'], report_cv['weighted avg']['f1-score']]
        })
        
        comparison_path = os.path.join(output_dir, "baseline2_2_method_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison results saved: {comparison_path}")
        
        print("="*60)
        print(f"All results saved in '{output_dir}' directory")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
