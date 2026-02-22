# Codebook (English) — combined_data.csv

## File overview

- **Filename:** `combined_data.csv`
- **Rows:** 2,022
- **Columns:** 3
- **Text encoding:** UTF-8 
- **Language Note:** The Hashtag column may contain Korean (and occasionally other non-English) text, as the dataset was constructed to train and evaluate Korean-specialized multimodal models (e.g., koCLIP).

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

### 4) `Korean–English Hashtag Glossary (Frequently Appearing Terms)`

Below are English equivalents for commonly appearing Korean hashtags in the dataset:

| Korean / Mixed Hashtag | English Equivalent | Notes |
|------------------------|-------------------|-------|
| 패션 | Fashion | — |
| 가을 | Autumn | Seasonal hashtag |
| 영화 | Movie | — |
| 올리브영 | Olive Young | Major Korean beauty retail brand |
| 인공지능 | Artificial Intelligence | Often appears with AI-related posts |
| 맛집 | Popular restaurant / Good restaurant | Common Korean food-related hashtag |
| 제로칼로리 | Zero calorie | Health / diet context |
| 일상 | Daily life | Lifestyle-related |
| 유머 | Humor | Comedy-related posts |
| 직장인 | Office worker | Work-life context |
| 데일리룩 | Daily outfit | Fashion styling |
| 오오티디 | OOTD (Outfit Of The Day) | Korean phonetic spelling of OOTD |
| 국내여행 | Domestic travel | Travel within Korea |
| 여행 | Travel | General travel context |
| 여자코디 | Women’s outfit styling | Fashion coordination |
| 아이브 | IVE | K-pop girl group |
| 웃긴짤 | Funny meme | Internet meme content |
| 패션템 | Fashion item | — |
| 패션템추천 | Fashion item recommendation | Product suggestion context |
| 데일리코디 | Daily outfit coordination | Styling-related |
| ファッション | Fashion | Japanese spelling |

The glossary lists frequently appearing Korean and mixed-language hashtags in the dataset. 
English equivalents are provided for clarity; however, translations are approximate and context-dependent.

## Notes for reproducibility

- Each image can appear in multiple rows (one row per hashtag).
- If you create train/validation splits, stratify by `Score` (or `Score_normalized`) to preserve the label distribution.
