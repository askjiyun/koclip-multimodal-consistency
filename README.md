# koclip-multimodal-consistency

Reproducible baselines for **Korean image–hashtag consistency** using **CLIP/koCLIP**.  
This repository accompanies the paper:

> **Predicting Semantic Consistency between Images and Korean Hashtags on Instagram**
> Jiyoon Oh & Jangmin Oh, School of AI Convergence, Sungshin Women's University

---

## Table of Contents

- [Description](#description)
- [Repository Structure](#repository-structure)
- [Dataset Information](#dataset-information)
- [Code Information](#code-information)
- [Requirements](#requirements)
- [Usage Instructions](#usage-instructions)
- [Methodology](#methodology)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Description

This repository investigates semantic consistency modeling between images and Korean hashtags, addressing the limitations of generic multimodal models in social media environments.
On image-centric social media platforms such as Instagram, user-generated hashtags often reflect subjective feelings, slang, or abstract concepts rather than literal descriptions of visual content. This creates a substantial semantic gap that general-purpose multi-modal models (e.g., CLIP) struggle to capture.

This project presents training and evaluation scripts for a **five-class semantic consistency prediction task** (scores 1–5), which quantifies the alignment between images and Korean hashtags.

We compare three progressive training strategies:
1. Similarity-based baselines  
2. Frozen-backbone classifiers  
3. End-to-end fine-tuning  

Experiments are conducted using both CLIP and KoCLIP backbones.

---
## Repository Structure

```
koclip-multimodal-consistency/
README.md
requirements.txt
data/
combined_data.csv #Annotated image-hashtag pairs
categorical_codebook.csv # Mapping table for categorical labels (e.g., Score 1–5 definitions)
DATA_CODEBOOK.md # Full English data dictionary describing variables and label schema
model/
baseline/
baseline_1_1_clip.py # CLIP similarity baseline 
baseline_1_2_koclip.py # koCLIP similarity baseline 
baseline_2_1_clip.py # CLIP fine-tuning (frozen CLIP + classifier head)
baseline_2_2_koclip.py # koCLIP fine-tuning (frozen koCLIP + classifier head)
FT_clip.py # CLIP end-to-end fine-tuning
FT_koclip.py # KoCLIP end-to-end fine-tuning
```
---
## Requirments
Create a virtual environment (macOS):

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Minimal requirements.txt:
```
torch
transformers
pandas
numpy
pillow
tqdm
scikit-learn
matplotlib
seaborn
```

---
## Dataset Information
### Overview

The dataset consists of approximately **2,000 image–hashtag pairs** collected from public Korean Instagram posts. 

Posts were gathered across **nine hashtag categories** representing lifestyle, technology, and seasonal trends:

| Korean Hashtag | English Translation |
|---|---|
| #국내여행 | Domestic Travel |
| #ai | AI |
| #가을 | Autumn |
| #제로칼로리 | Zero Calories |
| #맛집 | Good Restaurant |
| #영화 | Movie |
| #올리브영 | Olive Young (cosmetics brand) |
| #패션 | Fashion |
| #iphone16 | iPhone 16 |

For each of the nine seed hashtags, the 20 most recent public posts were collected using a Python Selenium-based web crawler, resulting in 180 unique posts. 

From each post, the primary image and all associated hashtags were extracted. After removing duplicates and invalid entries, this process yielded approximately 2,000 valid image–hashtag pairs.

---

### Data File Format

The main data file `data/combined_data.csv` contains the following columns:

| Column | Type | Description |
|---|---|---|
| `Image Filename` | string | Filename of the image (e.g., `abc.jpg`). Identical numeric identifiers correspond to the same image instance. |
| `Hashtag` | string | Korean hashtag text associated with the image |
| `Score` | integer (1–5) | Semantic consistency score between the image and hashtag |

---

### Consistency Score Codebook

Each image–hashtag pair is assigned a semantic consistency score on a **five-point Likert scale**, annotated using OpenAI GPT-4 Structured Output.

| Score | Label | Description |
|---|---|---|
| 1 | Very Low | The hashtag has no semantic connection to the image content |
| 2 | Low | The hashtag has minimal or tangential relevance to the image |
| 3 | Medium | The hashtag is partially related but not strongly aligned with the image |
| 4 | High | The hashtag is clearly relevant and semantically consistent with the image |
| 5 | Very High | The hashtag perfectly describes or complements the image content |

To assess reliability, a subset of samples was manually validated. The agreement between human annotations and GPT-based labels achieved **Cohen's Kappa (κ = 0.68)**, indicating substantial agreement.

---

### Image Files

Due to copyright and privacy considerations, the raw image files are not directly included in this repository.

The image dataset can be accessed via the following Google Drive link:

👉 **Image Dataset (Google Drive)**  
https://drive.google.com/drive/folders/1XS3ti5qAMHS4pXVlr4gFDfX6M0TO5acn?usp=drive_link

After downloading, place the image files under: `data/image/` and and update the image path in the training scripts accordingly.

---

## Code Information

All scripts are located in `model/baseline/` and implement three progressive training strategies:

| Script | Model | Strategy | Description |
|---|---|---|---|
| `baseline_1_1_clip.py` | CLIP | Similarity-based | Cosine similarity with fixed thresholds |
| `baseline_1_2_koclip.py` | KoCLIP | Similarity-based | Cosine similarity with fixed thresholds |
| `baseline_2_1_clip.py` | CLIP | Frozen backbone + classifier | Frozen encoder with trainable MLP head |
| `baseline_2_2_koclip.py` | KoCLIP | Frozen backbone + classifier | Frozen encoder with trainable MLP head |
| `FT_clip.py` | CLIP | End-to-end fine-tuning | Full model fine-tuning with classifier head |
| `FT_koclip.py` | KoCLIP | End-to-end fine-tuning | Full model fine-tuning with classifier head |

---

## Requirements

**Python 3.11** is recommended. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Dependencies

```
torch
transformers
pandas
numpy
pillow
tqdm
scikit-learn
matplotlib
seaborn
```

### Hardware

All experiments were conducted on an **NVIDIA TITAN RTX GPU (24 GB VRAM)**. Classification-based models were trained for up to 20 epochs with a batch size of 32.

---

## Usage Instructions

1. **Prepare the dataset**: Place your images under `data/image/` and ensure `data/combined_data.csv` contains the correct filenames.

2. **Adjust file paths**: Edit the data paths inside each script to match your local directory structure.

3. **Run a baseline**:

```bash
# Similarity baseline (threshold search)
python model/baseline/baseline_1_2_koclip.py

# Frozen backbone + classifier
python model/baseline/baseline_2_2_koclip.py

# End-to-end fine-tuning
python model/baseline/FT_koclip.py
```

4. **Results**:
   - Dataset split: 80/20 (train/test)
   - Metrics printed: Accuracy, Precision, Recall, F1-score
   - Confusion matrices saved as image files
   - Training curves exported as training_results.png
   - Fine-tuned weights saved as .pth files
---

## Methodology

1. **Data Collection**: Korean Instagram posts collected via Selenium-based web crawler across nine hashtag categories.
2. **Automated Labeling**: GPT-4 Structured Output was used to assign five-level semantic consistency scores. Human validation achieved Cohen’s Kappa (κ = 0.68), indicating substantial agreement.
3. **Model Training**: Three progressive strategies evaluate similarity-based baselines, frozen backbone with trainable classifier head, and end-to-end fine-tuning using both CLIP (`clip-vit-base-patch32`) and KoCLIP (`Bingsu/clip-vit-large-patch14-ko`) backbones.
4. **Evaluation**: 80/20 train/test split with cross-entropy loss, Adam optimizer (lr = 5e-5), and early stopping.

For full details, please refer to the accompanying paper.

---

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{oh2025koclip,
  author  = {Oh, Jiyoon and Oh, Jangmin},
  title   = {Predicting Semantic Consistency between Images and Korean Hashtags on Instagram},
  journal = {PeerJ Computer Science},
  year    = {2025},
  url     = {https://github.com/askjiyun/koclip-multimodal-consistency}
}
```

--- 

## Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (`openai/clip-vit-base-patch32`)
- [KoCLIP](https://huggingface.co/Bingsu/clip-vit-large-patch14-ko) (`Bingsu/clip-vit-large-patch14-ko`)

---

## License

This project is released under the [MIT License](LICENSE).

