# koclip-multimodal-consistency

Reproducible baselines for **Korean image–hashtag consistency** using **CLIP/koCLIP**.  
Includes training & evaluation scripts for 5-class consistency prediction (0–4) with 80/20 splits. Metrics and artifacts are exported for paper reproduction.

> **TL;DR (KOR)**: CLIP/koCLIP으로 이미지–해시태그 정합성을 5단계로 분류/평가하는 재현 가능한 베이스라인 코드입니다.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Data Format (not included)](#data-format-not-included)
- [Running the Baselines](#running-the-baselines)
  - [Similarity Baselines](#1-similarity-baselines)
  - [Fine-tuning Baselines](#2-fine-tuning-baselines)
- [Reproducibility Notes](#reproducibility-notes)
- [Example Commands (macOS)](#example-commands-macos)
- [Metrics & Artifacts](#metrics--artifacts)
- [Citation](#citation)
- [License](#license)

---
## Repository Structure
> model/
> baseline/
> baseline_1_1_clip.py # CLIP similarity baseline 
> baseline_1_2_koclip.py # koCLIP similarity baseline 
> baseline_2_1_clip.py # CLIP fine-tuning (frozen CLIP + classifier head)
> baseline_2_2_koclip.py # koCLIP fine-tuning (frozen koCLIP + classifier head)
> FT_clip.py # Full fine-tuning / evaluation (CLIP)
> FT_koclip.py # Full fine-tuning / evaluation (koCLIP)
>
---
## Requirments
Create a virtual environment (macOS):

> python -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt


- Minimal requirements.txt:
> torch
> transformers
> pandas
> numpy
> pillow
> tqdm
> scikit-learn
> matplotlib
> seaborn

---
## Data Format 
> Insta Project/
>   data/
>     combined_data.csv
>     image/
>       <image files referenced by CSV>

- combined_data.csv columns:
  - Image Filename: e.g., abc.jpg
  - Hashtag: text string
  - Score: integer label (1–5) → scripts typically normalize to 0–4
- Images live under data/image/.
- The repository does not include data. Adjust paths in scripts (e.g.,path/yourpath/data/... → your local path).

---
## Example command (macOS)

> Edit paths inside the script or convert to CLI as needed.

> # Example: run a fine-tuning baseline (paths inside the script)
> python model/baseline/baseline_2_2_koclip.py

For similarity baselines (threshold search), ensure the CSV/images paths are correct, then:

> python model/baseline/baseline_1_2_koclip.py

--- 
## Citation

If you use this repository in academic work, please cite:

> @software{askjiyun_koclip_multimodal_consistency_2025,
>   author  = {Jiyoon, Jangmin},
>   title   = {koclip-multimodal-consistency: Reproducible baselines for Korean image–hashtag consistency},
>   year    = {2025},
>   url     = {https://github.com/askjiyun/koclip-multimodal-consistency}
> }

--- 
## Acknowledgments
- Hugging Face Transformers
- OpenAI CLIP (`openai/clip-vit-base-patch32`)
- koCLIP (`Bingsu/clip-vit-large-patch14-ko`)

