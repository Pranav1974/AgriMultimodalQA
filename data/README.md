# 📦 AgriMultimodalQA — Dataset Guide

This folder contains all training data for the AgriMultimodalQA system.

---

## 🗂 Folder Structure

```
data/
├── raw/
│   ├── images/
│   │   ├── plantvillage/       ← 54,306 leaf images (38 classes) [DOWNLOAD NEEDED]
│   │   └── rice_leaf/          ← 120 rice leaf images (4 classes) [DOWNLOAD NEEDED]
│   │
│   ├── nlp/
│   │   ├── intent_seed_data.json    ← Auto-created ✅
│   │   ├── ner_seed_data.json       ← Auto-created ✅
│   │   ├── english/intent_data.json ← Auto-created ✅
│   │   ├── tamil/intent_data.json   ← Auto-created ✅
│   │   ├── telugu/intent_data.json  ← Auto-created ✅
│   │   └── hindi/intent_data.json   ← Auto-created ✅
│   │
│   ├── weather/
│   │   └── nasa_power/         ← Pre-fetched regional weather JSON [Auto-fetch]
│   │
│   └── kg_sources/             ← Additional knowledge sources
│
├── processed/                  ← Generated during training (auto-created)
│   ├── multilingual_tokens/
│   ├── image_tensors/
│   ├── weather_vectors/
│   └── kg_triples/
│
└── embeddings/
    └── document_embeddings/    ← FAISS index (auto-created on first run)
```

---

## 📥 How to Download Each Dataset

### 1️⃣ PlantVillage Dataset (REQUIRED for Image Model)

> 54,306 leaf images across 38 classes (14 crop types)  
> Size: ~1.2 GB

**Step 1: Setup Kaggle API**
```
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save kaggle.json to: C:\Users\prana\.kaggle\kaggle.json
```

**Step 2: Download**
```bash
pip install kaggle
kaggle datasets download -d emmarex/plantdisease -p data/raw/images/plantvillage --unzip
```

**OR Manual Download:**
- Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
- Click Download → Unzip to `data/raw/images/plantvillage/`

**Expected structure after download:**
```
data/raw/images/plantvillage/
├── Apple___Apple_scab/          (630 images)
├── Apple___Black_rot/           (621 images)
├── Apple___Cedar_apple_rust/    (275 images)
├── Apple___healthy/             (1645 images)
├── Tomato___Early_blight/       (1000 images)
├── Tomato___Late_blight/        (1909 images)
├── Tomato___healthy/            (1591 images)
├── ...                          (38 folders total)
```

---

### 2️⃣ Rice Leaf Disease Dataset (REQUIRED for Rice-Specific Model)

> 120 images, 4 disease classes  
> Size: ~4 MB (very small!)

```bash
kaggle datasets download -d vbookshelf/rice-leaf-diseases -p data/raw/images/rice_leaf --unzip
```

**OR Manual:**
- Visit: https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
- Download → Unzip to `data/raw/images/rice_leaf/`

**Expected structure:**
```
data/raw/images/rice_leaf/
├── Bacterial Leaf Blight/    (40 images)
├── Brown Spot/               (40 images)
├── Leaf Smut/                (40 images)
```

---

### 3️⃣ NLP Data — ALREADY CREATED ✅

> These files were auto-generated:
> - `data/raw/nlp/intent_seed_data.json` — 55 labelled queries
> - `data/raw/nlp/ner_seed_data.json` — 5 annotated NER examples
> - `data/raw/nlp/english/intent_data.json`
> - `data/raw/nlp/tamil/intent_data.json`
> - `data/raw/nlp/hindi/intent_data.json`

**For a larger NLP dataset (recommended for paper):**

| Dataset | Link | Use |
|---------|------|-----|
| AgroQA | https://huggingface.co/datasets/cs231n/agricultural-qa | QA pairs |
| AI4Bharat Samanantar | https://ai4bharat.org/samanantar | Indian language |
| IndicNLP Corpus | https://github.com/ai4bharat/indicnlp_corpus | Tamil/Hindi/Telugu |

---

### 4️⃣ Weather Data — AUTO-FETCH via NASA POWER ✅

> No download needed! NASA POWER API is completely FREE.

```bash
# Pre-fetch weather for Indian regions
python data/download_datasets.py --weather
```

---

## ⚡ Quick Start (All at Once)

```bash
# Step 1: Create folders + NLP seed data (no internet needed)
python data/download_datasets.py --structure --nlp

# Step 2: Fetch weather data (needs internet, free)
python data/download_datasets.py --weather

# Step 3: Download image datasets (needs Kaggle account)
python data/download_datasets.py --rice         # Small (4MB)
python data/download_datasets.py --plantvillage # Large (1.2GB)

# Step 4: Run system (works even without image data!)
python quick_test.py
uvicorn api.main:app --reload
streamlit run frontend/app.py
```

---

## 🎯 Minimum Dataset to Get Started

You can run the full system with **ZERO downloads** using:
- ✅ Rule-based NLP (no training needed)
- ✅ Built-in Knowledge Graph (39 nodes, 48 edges)  
- ✅ NASA POWER Weather API (live, free)
- ✅ Flan-T5-small for generation (downloads ~300 MB automatically)
- ⚠️ Image model uses ImageNet pretrained (not fine-tuned, lower accuracy)

For fine-tuned models, download the datasets above and run:
```bash
python experiments/train_intent_classifier.py
python experiments/train_image_model.py --data_dir data/raw/images/plantvillage
```

---

## 📊 Dataset Statistics

| Dataset | Samples | Classes | Size | Status |
|---------|---------|---------|------|--------|
| PlantVillage | 54,306 images | 38 | 1.2 GB | Download needed |
| Rice Leaf | 120 images | 4 | 4 MB | Download needed |
| Intent (seed) | 55 queries | 10 intents | <1 MB | Auto-created ✅ |
| NER (seed) | 5 sentences | 7 types | <1 MB | Auto-created ✅ |
| Weather | Live API | - | - | Auto-fetch ✅ |

---

## 🔗 All Dataset Links

| Dataset | URL |
|---------|-----|
| PlantVillage (Kaggle) | https://www.kaggle.com/datasets/emmarex/plantdisease |
| Rice Leaf Disease | https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases |
| Rice Leaf (alt.) | https://www.kaggle.com/datasets/nizorogbezuode/rice-leaf-diseases |
| AgroQA | https://huggingface.co/datasets |
| NASA POWER API | https://power.larc.nasa.gov/api |
| AI4Bharat NLP | https://ai4bharat.org |
| IndicNLP Corpus | https://github.com/ai4bharat/indicnlp_corpus |
| Samanantar (translation) | https://ai4bharat.org/samanantar |
