# Codebook (English) — combined_data.csv

## File overview

- **Filename:** `combined_data.csv`
- **Rows:** 2,022
- **Columns:** 3
- **Text encoding:** UTF-8 (the `Hashtag` column contains Korean text)

## Variables (data dictionary)

### 1) `Image Filename`

- **Type:** string
- **Description:** File name of the Instagram post image.  
  Each image is assigned a numeric identifier so that multiple hashtags from a single post can be represented as separate image–hashtag pairs.  
  Filenames with the same numeric identifier indicate the same image instance.
- **Example:** `fall_01.png`

### 2) `Hashtag`

- **Type:** string
- **Description:** A single hashtag text associated with the image.
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

## Notes for reproducibility

- Each image can appear in multiple rows (one row per hashtag).
- If you create train/validation splits, stratify by `Score` (or `Score_normalized`) to preserve the label distribution.
