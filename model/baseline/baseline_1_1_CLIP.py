#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline CLIP Model Evaluation Script

This script uses CLIP (openai/clip-vit-base-patch32) to predict and evaluate 
the consistency between Instagram images and hashtags into five classes (0–4).  
It loads a CSV file (image filename, hashtag, score), splits the data into 
an 8:2 train/validation set, and extracts image–text embeddings for each sample 
to compute cosine similarity.

Two methods are applied to map similarity scores into classes:
1) Threshold-based: Four thresholds are randomly sampled and combined, and the 
   best boundary set (based on macro-F1) is selected to map scores into five levels.  
2) Simple mapping: A fixed-interval mapping is applied, dividing similarity 
   values into five classes at 0.2 intervals.

Evaluation metrics include MAE, Accuracy, classification reports, and confusion 
matrices. Intermediate results (similarities/labels as npy files), the best 
thresholds (JSON), and summary outputs (JSON) are also stored.  
If a GPU is available, CUDA will be used automatically.


"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    classification_report, mean_absolute_error, accuracy_score, 
    f1_score, confusion_matrix
)
from transformers import CLIPProcessor, CLIPModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def setup_environment():
    """settings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model and processor loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return device, model, processor


def load_and_prepare_data(data_path="path/to/your/combined_data.csv"):
    """Data Load & Preparation"""
    try:
        df = pd.read_csv(data_path)
        df['Score'] = df['Score'] - 1  # Adjust the score to 0-4
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Score distribution:\n{df['Score'].value_counts().sort_index()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def get_clip_embeddings(image_path, text, processor, model, device):
    """CLIP Embedding Extraction Function"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        image_embed = outputs.image_embeds[0].cpu().numpy()
        text_embed = outputs.text_embeds[0].cpu().numpy()
        
        return image_embed, text_embed
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def compute_similarities(df_subset, processor, model, device, image_dir="/path/to/your/image/folder"):
    """Similarity and Score Extraction"""
    similarities = []
    labels = []
    failed_count = 0
    
    print(f"Processing {len(df_subset)} samples...")
    
    for i, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Computing similarities"):
        try:
            image_path = os.path.join(image_dir, row['Image Filename'])
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                failed_count += 1
                continue
                
            hashtag = row['Hashtag']
            label = row['Score']

            img_embed, txt_embed = get_clip_embeddings(
                image_path, hashtag, processor, model, device
            )
            
            if img_embed is None or txt_embed is None:
                failed_count += 1
                continue
            
            sim = cosine_similarity([img_embed], [txt_embed])[0][0]
            similarities.append(sim)
            labels.append(label)

            if (i + 1) % 100 == 0:
                print(f"[{i+1}] Hashtag: {hashtag[:50]}... | Score: {label} | Similarity: {sim:.4f}")
                
        except Exception as e:
            print(f"[{i}] Error: {str(e)}")
            failed_count += 1
            continue
            
    print(f"Processed: {len(similarities)} samples, Failed: {failed_count} samples")
    return similarities, labels


def find_optimal_multiclass_thresholds(similarities, scores, n_combinations=5000):
    """Finding the Optimal Threshold for Multi-Class Classification"""
    print("Finding optimal multiclass thresholds...")
    
    # Generate threshold candidate combinations
    np.random.seed(42)
    candidate_thresholds = sorted(np.random.uniform(0.05, 0.95, 100))
    candidate_combinations = list(combinations(candidate_thresholds, 4))
    
    best_f1 = -1
    best_thresholds = None
    
    print(f"Testing {min(n_combinations, len(candidate_combinations))} threshold combinations...")
    
    for i, t in enumerate(candidate_combinations[:n_combinations]):
        if i % 1000 == 0:
            print(f"Progress: {i}/{n_combinations}")
            
        preds = [similarity_to_score_with_threshold(s, t) for s in similarities]
        f1 = f1_score(scores, preds, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = t
    
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Best thresholds: {best_thresholds}")
    return best_thresholds


def similarity_to_score_with_threshold(similarity, thresholds):
    """Similarity-to-score conversion using thresholds"""
    if similarity < thresholds[0]:
        return 0
    elif similarity < thresholds[1]:
        return 1
    elif similarity < thresholds[2]:
        return 2
    elif similarity < thresholds[3]:
        return 3
    else:
        return 4


def similarity_to_score_simple(sim):
    """Simple similarity-to-score mapping"""
    if sim < 0.2:
        return 0
    elif sim < 0.4:
        return 1
    elif sim < 0.6:
        return 2
    elif sim < 0.8:
        return 3
    else:
        return 4


def evaluate_and_save(name, y_true, y_pred, output_dir="./results/", thresholds=None):
    """Evaluation and result saving"""
    os.makedirs(output_dir, exist_ok=True)
    
    mae = mean_absolute_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    
    conf_filename = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.savefig(conf_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {conf_filename}")

    # Save text report
    txt_filename = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_results.txt")
    with open(txt_filename, "w", encoding='utf-8') as f:
        f.write(f"Baseline 1-1 Evaluation Result: {name}\n\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        if thresholds is not None:
            f.write(f"Best Thresholds: {thresholds}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Results saved: {txt_filename}")

    # Save CSV file
    csv_filename = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_metrics.csv")
    summary_df = pd.DataFrame({
        "Method": [name],
        "MAE": [mae],
        "Accuracy": [acc],
        "Macro_F1": [report_dict['macro avg']['f1-score']],
        "Weighted_F1": [report_dict['weighted avg']['f1-score']]
    })
    summary_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved: {csv_filename}")
    
    return mae, acc, report_dict


def save_embeddings(train_sims, train_scores, val_sims, val_scores, output_dir="./results/"):
    """Save embedding results"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "train_similarities.npy"), np.array(train_sims))
    np.save(os.path.join(output_dir, "train_scores.npy"), np.array(train_scores))
    np.save(os.path.join(output_dir, "val_similarities.npy"), np.array(val_sims))
    np.save(os.path.join(output_dir, "val_scores.npy"), np.array(val_scores))
    
    print("Embeddings saved successfully")


def main():
    print("=" * 60)
    print("Baseline CLIP Model Evaluation")
    print("=" * 60)
    
    try:
        # 1. Environment Setup
        device, model, processor = setup_environment()
        
        # 2. Load data
        df = load_and_prepare_data()
        
        # 3. train/valiation split
        train_df, val_df = train_test_split(
            df, stratify=df['Score'], test_size=0.2, random_state=42
        )
        print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        # 4. Process training data
        print("\n" + "="*40)
        print("Processing Training Data")
        print("="*40)
        train_sims, train_scores = compute_similarities(train_df, processor, model, device)
        
        # 5. Process validation data
        print("\n" + "="*40)
        print("Processing Validation Data") 
        print("="*40)
        val_sims, val_scores = compute_similarities(val_df, processor, model, device)
        
        # 6. Save embeddings
        save_embeddings(train_sims, train_scores, val_sims, val_scores)
        
        # 7. Find optimal thresholds for multiclass classification
        print("\n" + "="*40)
        print("Finding Optimal Multiclass Thresholds")
        print("="*40)
        best_thresholds = find_optimal_multiclass_thresholds(train_sims, train_scores)
        
        # 8. Evaluate and save results
        print("\n" + "="*40)
        print("Multiclass Evaluation Results")
        print("="*40)
        
        # Threshold-based prediction
        val_pred_scores_thresh = [similarity_to_score_with_threshold(s, best_thresholds) for s in val_sims]
        mae1, acc1, _ = evaluate_and_save(
            name="Baseline 1-1 Threshold-Based",
            y_true=val_scores,
            y_pred=val_pred_scores_thresh,
            thresholds=best_thresholds
        )
        
        # Simple-mapping prediction
        val_pred_scores_simple = [similarity_to_score_simple(sim) for sim in val_sims]
        mae2, acc2, _ = evaluate_and_save(
            name="Baseline 1-1 Simple-Mapping",
            y_true=val_scores,
            y_pred=val_pred_scores_simple
        )
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Binary Classification Accuracy: {np.mean([p == t for p, t in zip(binary_preds, binary_labels)]):.4f}")
        print(f"Threshold-Based Method: MAE={mae1:.4f}, Accuracy={acc1:.4f}")
        print(f"Simple-Mapping Method:  MAE={mae2:.4f}, Accuracy={acc2:.4f}")
        print("="*60)
        print("All results saved in './results/' directory")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
