# Codebook (English) — combined_data.csv

## File overview

- **Filename:** `combined_data.csv`
- **Rows:** 2,022
- **Columns:** 3
- **Text encoding:** UTF-8 
- **Language Note:**  
  The `Hashtag` column may contain Korean and other non-English text, as the dataset was constructed to train and evaluate Korean-specialized multimodal models (e.g., koCLIP).  
  To improve accessibility for editors, reviewers, and readers, an additional column `Hashtag_en` has been provided. This column contains English translations of the original hashtags for reference purposes.

## Variables (data dictionary)

### 1) `Image Filename`

- **Type:** string
- **Description:** File name of the Instagram post image.  
  Each image is assigned a numeric identifier so that multiple hashtags from a single post can be represented as separate image–hashtag pairs.  
  Filenames with the same numeric identifier indicate the same image instance.
- **Example:** `fall_01.png`

### 2) `Hashtag`

- **Type:** string
- **Description:** A single hashtag associated with the image.

The hashtag text may include Korean words, slang, neologisms, or mixed-language expressions. This reflects real-world Instagram usage and was intentionally preserved to support training and evaluation of Korean-language multimodal models.

English translations of frequently used Korean hashtags are provided in the glossary section below.
- **Example:** `맛집(Restaurant)`, `오운완(WorkoutDone)`, `여행(Travel)`

### 3) `Score`

- **Type:** integer (ordinal categorical label)
- **Description:** GPT 4-annotated semantic consistency level between the image and the hashtag.
- **Original label range:** 1–5
- **Meaning (original 1–5):**
  - 1 = Very low relevance
  - 2 = Low relevance
  - 3 = Medium relevance
  - 4 = High relevance
  - 5 = Very high relevance
- **Common preprocessing used in code:** `Score_normalized = Score - 1` (maps 1–5 → 0–4)

### 4) `Hashtag_en`

- **Type:** string
- **Description:**  
  English translation of the original `Hashtag` column.

  This column is provided for interpretability and transparency so that non-Korean-speaking readers can understand the semantic meaning of each hashtag.

  **Important:**  
  The `Hashtag_en` column is intended for reference and documentation purposes only.  
  The experimental models described in the paper were trained using the original `Hashtag` column.  
  Therefore, researchers reproducing the experiments may use the dataset exactly as originally structured without modification.

- **Example:**  
  `맛집` → `Popular restaurant`  
  `가을` → `Autumn`

## Notes for reproducibility

- Each image can appear in multiple rows (one row per hashtag).
- If you create train/validation splits, stratify by `Score` (or `Score_normalized`) to preserve the label distribution.
- The original experiments were conducted using the `Hashtag` column.  
  The `Hashtag_en` column does not affect the model architecture or training pipeline and is included solely for interpretability.
