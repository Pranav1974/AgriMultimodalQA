"""
download_datasets.py
====================
Downloads all required datasets for AgriMultimodalQA.

Datasets:
  1. PlantVillage   - 54,306 leaf images, 38 classes (Image model)
  2. Rice Leaf      - 120 images, 4 disease classes (Image model)
  3. AgroQA NLP     - Agricultural QA pairs (NLP training)
  4. iNaturalist    - Additional plant images

Run:
    python data/download_datasets.py --all
    python data/download_datasets.py --plantvillage
    python data/download_datasets.py --rice
    python data/download_datasets.py --nlp
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "raw"
NLP_DIR  = RAW_DIR  / "nlp"
IMG_DIR  = RAW_DIR  / "images"
WX_DIR   = RAW_DIR  / "weather"

# ── Try imports ───────────────────────────────────────────────
try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    import gdown
    _GDOWN = True
except ImportError:
    _GDOWN = False
    logger.warning("gdown not installed. Run: pip install gdown")

try:
    import kaggle
    _KAGGLE = True
except ImportError:
    _KAGGLE = False


# ═══════════════════════════════════════════════════════════════
#  DATASET 1 — PlantVillage (Kaggle)
# ═══════════════════════════════════════════════════════════════
def download_plantvillage():
    """
    PlantVillage Dataset — 54,306 leaf images, 38 classes.

    Source: https://www.kaggle.com/datasets/emmarex/plantdisease
    Size  : ~1.2 GB

    Requires Kaggle API key:
      1. Go to https://www.kaggle.com/account
      2. Click "Create New API Token"
      3. Save kaggle.json to C:\\Users\\YourName\\.kaggle\\kaggle.json
    """
    dest = IMG_DIR / "plantvillage"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("Downloading PlantVillage Dataset (~1.2 GB)")
    logger.info("=" * 55)

    if _KAGGLE:
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "emmarex/plantdisease",
                path=str(dest),
                unzip=True
            )
            logger.info(f"PlantVillage downloaded to: {dest}")
            return True
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")

    # Fallback: manual instructions
    logger.warning("""
    ─────────────────────────────────────────────────────
    MANUAL DOWNLOAD — PlantVillage:
    
    Option A (Kaggle CLI — recommended):
      1. Install: pip install kaggle
      2. Get API key from: https://www.kaggle.com/account
         → Download kaggle.json
         → Place in: C:\\Users\\prana\\.kaggle\\kaggle.json
      3. Run:
         kaggle datasets download -d emmarex/plantdisease
         -p data\\raw\\images\\plantvillage --unzip

    Option B (Browser download):
      1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
      2. Click "Download"
      3. Unzip to: data\\raw\\images\\plantvillage\\
    
    Expected structure after unzip:
      data/raw/images/plantvillage/
        ├── Apple___Apple_scab/       (images/*.JPG)
        ├── Apple___healthy/
        ├── Tomato___Early_blight/
        ├── ...                       (38 folders total)
    ─────────────────────────────────────────────────────
    """)
    return False


# ═══════════════════════════════════════════════════════════════
#  DATASET 2 — Rice Leaf Disease (Kaggle)
# ═══════════════════════════════════════════════════════════════
def download_rice_leaf():
    """
    Rice Leaf Disease Dataset — 120 images, 4 classes.

    Source: https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
    Size  : ~4 MB (very small, fast download)
    
    Classes:
      - Bacterial Leaf Blight
      - Brown Spot
      - Healthy
      - Leaf Smut
    """
    dest = IMG_DIR / "rice_leaf"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("Downloading Rice Leaf Disease Dataset (~4 MB)")
    logger.info("=" * 55)

    if _KAGGLE:
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "vbookshelf/rice-leaf-diseases",
                path=str(dest),
                unzip=True
            )
            logger.info(f"Rice Leaf downloaded to: {dest}")
            return True
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")

    logger.warning("""
    ─────────────────────────────────────────────────────
    MANUAL DOWNLOAD — Rice Leaf Disease:
    
    Option A (Kaggle CLI):
      kaggle datasets download -d vbookshelf/rice-leaf-diseases
      -p data\\raw\\images\\rice_leaf --unzip
    
    Option B (Browser):
      https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
    
    Expected structure:
      data/raw/images/rice_leaf/
        ├── Bacterial Leaf Blight/    (40 images)
        ├── Brown Spot/               (40 images)
        ├── Leaf Smut/                (40 images)
    ─────────────────────────────────────────────────────
    """)
    return False


# ═══════════════════════════════════════════════════════════════
#  DATASET 3 — NLP Data (Multilingual Agricultural)
# ═══════════════════════════════════════════════════════════════
def create_nlp_dataset():
    """
    Create NLP training data for intent + NER.

    Sources used:
      1. Built-in seed data (already in experiments/train_intent_classifier.py)
      2. Vikaspedia scrape stubs (agricultural info in Hindi/Tamil/Telugu)
      3. ICAR (Indian Council of Agricultural Research) guidelines

    This function creates the SEED data files.
    For full training, use the larger datasets linked below.
    """
    logger.info("=" * 55)
    logger.info("Creating NLP Dataset (Seed Data)")
    logger.info("=" * 55)

    # ── Intent Classification Seed Data ──────────────────────
    intent_seed = [
        # DISEASE_DIAGNOSIS
        {"text": "My rice plants have yellow spots on leaves", "label": "DISEASE_DIAGNOSIS", "lang": "en"},
        {"text": "Yellow leaves on paddy plant", "label": "DISEASE_DIAGNOSIS", "lang": "en"},
        {"text": "Brown lesions on wheat leaves", "label": "DISEASE_DIAGNOSIS", "lang": "en"},
        {"text": "White patches on my crop leaves", "label": "DISEASE_DIAGNOSIS", "lang": "en"},
        {"text": "Tomato leaves turning black", "label": "DISEASE_DIAGNOSIS", "lang": "en"},
        {"text": "What is wrong with my rice plant", "label": "DISEASE_DIAGNOSIS", "lang": "en"},
        {"text": "நெல் இலைகளில் மஞ்சள் புள்ளிகள்", "label": "DISEASE_DIAGNOSIS", "lang": "ta"},
        {"text": "வரி செடிகளில் பழுப்பு நோய்", "label": "DISEASE_DIAGNOSIS", "lang": "ta"},
        {"text": "నా వరి ఆకులపై పసుపు మచ్చలు ఉన్నాయి", "label": "DISEASE_DIAGNOSIS", "lang": "te"},
        {"text": "धान की पत्तियों पर पीले धब्बे हैं", "label": "DISEASE_DIAGNOSIS", "lang": "hi"},
        {"text": "गेहूं के पत्तों पर भूरे धब्बे", "label": "DISEASE_DIAGNOSIS", "lang": "hi"},

        # TREATMENT_ADVICE
        {"text": "How to treat rice blast disease", "label": "TREATMENT_ADVICE", "lang": "en"},
        {"text": "What spray should I use for yellow spots", "label": "TREATMENT_ADVICE", "lang": "en"},
        {"text": "Treatment for bacterial leaf blight in paddy", "label": "TREATMENT_ADVICE", "lang": "en"},
        {"text": "How to cure tomato blight", "label": "TREATMENT_ADVICE", "lang": "en"},
        {"text": "Best fungicide for wheat rust", "label": "TREATMENT_ADVICE", "lang": "en"},
        {"text": "நெல் தடுப்பு மருந்து என்ன", "label": "TREATMENT_ADVICE", "lang": "ta"},
        {"text": "వరి తెగులుకు ఏ మందు వేయాలి", "label": "TREATMENT_ADVICE", "lang": "te"},
        {"text": "धान के रोग का उपचार क्या है", "label": "TREATMENT_ADVICE", "lang": "hi"},

        # FERTILIZER_QUERY
        {"text": "What fertilizer should I use for rice", "label": "FERTILIZER_QUERY", "lang": "en"},
        {"text": "How much urea per acre for paddy", "label": "FERTILIZER_QUERY", "lang": "en"},
        {"text": "NPK ratio for wheat crop", "label": "FERTILIZER_QUERY", "lang": "en"},
        {"text": "Which fertilizer is best for cotton", "label": "FERTILIZER_QUERY", "lang": "en"},
        {"text": "DAP dosage for rice", "label": "FERTILIZER_QUERY", "lang": "en"},
        {"text": "நெல்லுக்கு என்ன உரம் போட வேண்டும்", "label": "FERTILIZER_QUERY", "lang": "ta"},
        {"text": "धान के लिए खाद की मात्रा", "label": "FERTILIZER_QUERY", "lang": "hi"},

        # WEATHER_ADVICE
        {"text": "Is this weather good for sowing", "label": "WEATHER_ADVICE", "lang": "en"},
        {"text": "Will rain affect my crops this week", "label": "WEATHER_ADVICE", "lang": "en"},
        {"text": "When should I sow wheat based on temperature", "label": "WEATHER_ADVICE", "lang": "en"},
        {"text": "Is humidity high enough for paddy growth", "label": "WEATHER_ADVICE", "lang": "en"},

        # PEST_CONTROL
        {"text": "How to control aphids in cotton", "label": "PEST_CONTROL", "lang": "en"},
        {"text": "Stem borer treatment in rice", "label": "PEST_CONTROL", "lang": "en"},
        {"text": "Pesticide for rice planthopper", "label": "PEST_CONTROL", "lang": "en"},
        {"text": "How to kill whiteflies on tomato", "label": "PEST_CONTROL", "lang": "en"},
        {"text": "Brown planthopper management in paddy", "label": "PEST_CONTROL", "lang": "en"},

        # SOIL_QUERY
        {"text": "What soil pH is needed for rice cultivation", "label": "SOIL_QUERY", "lang": "en"},
        {"text": "Best soil type for cotton farming", "label": "SOIL_QUERY", "lang": "en"},
        {"text": "How to improve soil drainage for paddy", "label": "SOIL_QUERY", "lang": "en"},

        # IRRIGATION_QUERY
        {"text": "How often should I water rice plants", "label": "IRRIGATION_QUERY", "lang": "en"},
        {"text": "Drip irrigation schedule for cotton", "label": "IRRIGATION_QUERY", "lang": "en"},
        {"text": "Water requirement for wheat per day", "label": "IRRIGATION_QUERY", "lang": "en"},

        # CROP_RECOMMENDATION
        {"text": "Which crop should I grow in summer", "label": "CROP_RECOMMENDATION", "lang": "en"},
        {"text": "Best crop for Tamil Nadu monsoon season", "label": "CROP_RECOMMENDATION", "lang": "en"},
        {"text": "What to plant this kharif season", "label": "CROP_RECOMMENDATION", "lang": "en"},
        {"text": "Profitable crop for black soil in Maharashtra", "label": "CROP_RECOMMENDATION", "lang": "en"},

        # GENERAL_AGRI_INFO
        {"text": "Tell me about paddy cultivation methods", "label": "GENERAL_AGRI_INFO", "lang": "en"},
        {"text": "What is integrated pest management", "label": "GENERAL_AGRI_INFO", "lang": "en"},
        {"text": "Explain crop rotation benefits", "label": "GENERAL_AGRI_INFO", "lang": "en"},
        {"text": "How to improve farm yield", "label": "GENERAL_AGRI_INFO", "lang": "en"},

        # OUT_OF_SCOPE
        {"text": "What is the capital of India", "label": "OUT_OF_SCOPE", "lang": "en"},
        {"text": "Tell me a joke", "label": "OUT_OF_SCOPE", "lang": "en"},
        {"text": "Latest cricket score", "label": "OUT_OF_SCOPE", "lang": "en"},
        {"text": "How to cook biryani", "label": "OUT_OF_SCOPE", "lang": "en"},
    ]

    # ── NER Seed Data ─────────────────────────────────────────
    ner_seed = [
        {
            "text": "My rice plants in Tamil Nadu have yellow spots",
            "entities": [
                {"start": 3, "end": 7, "label": "CROP", "text": "rice"},
                {"start": 19, "end": 29, "label": "REGION", "text": "Tamil Nadu"},
                {"start": 35, "end": 47, "label": "SYMPTOM", "text": "yellow spots"},
            ]
        },
        {
            "text": "Apply carbendazim 50g per acre for wheat blast",
            "entities": [
                {"start": 6, "end": 17, "label": "CHEMICAL", "text": "carbendazim"},
                {"start": 18, "end": 21, "label": "QUANTITY", "text": "50g"},
                {"start": 22, "end": 29, "label": "QUANTITY", "text": "per acre"},
                {"start": 34, "end": 39, "label": "CROP", "text": "wheat"},
                {"start": 40, "end": 45, "label": "DISEASE", "text": "blast"},
            ]
        },
        {
            "text": "Cotton in Punjab has aphid infestation during kharif season",
            "entities": [
                {"start": 0, "end": 6, "label": "CROP", "text": "Cotton"},
                {"start": 10, "end": 16, "label": "REGION", "text": "Punjab"},
                {"start": 21, "end": 26, "label": "PEST", "text": "aphid"},
                {"start": 46, "end": 52, "label": "TIME", "text": "kharif"},
            ]
        },
        {
            "text": "Rice blast disease treated with tricyclazole 0.6g per liter",
            "entities": [
                {"start": 0, "end": 4, "label": "CROP", "text": "Rice"},
                {"start": 5, "end": 10, "label": "DISEASE", "text": "blast"},
                {"start": 32, "end": 43, "label": "CHEMICAL", "text": "tricyclazole"},
                {"start": 44, "end": 49, "label": "QUANTITY", "text": "0.6g"},
            ]
        },
        {
            "text": "Paddy leaves have brown spots due to high humidity in June",
            "entities": [
                {"start": 0, "end": 5, "label": "CROP", "text": "Paddy"},
                {"start": 18, "end": 29, "label": "SYMPTOM", "text": "brown spots"},
                {"start": 53, "end": 57, "label": "TIME", "text": "June"},
            ]
        },
    ]

    # Save files
    import json

    # Intent data
    for lang in ["english", "tamil", "telugu", "hindi"]:
        lang_dir = NLP_DIR / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

    intent_path = NLP_DIR / "intent_seed_data.json"
    with open(intent_path, "w", encoding="utf-8") as f:
        json.dump(intent_seed, f, ensure_ascii=False, indent=2)
    logger.info(f"Intent seed data saved: {intent_path} ({len(intent_seed)} samples)")

    ner_path = NLP_DIR / "ner_seed_data.json"
    with open(ner_path, "w", encoding="utf-8") as f:
        json.dump(ner_seed, f, ensure_ascii=False, indent=2)
    logger.info(f"NER seed data saved: {ner_path} ({len(ner_seed)} samples)")

    # Split by language
    for lang_code, lang_folder in [("en", "english"), ("ta", "tamil"), ("te", "telugu"), ("hi", "hindi")]:
        lang_data = [item for item in intent_seed if item.get("lang") == lang_code]
        lang_file = NLP_DIR / lang_folder / "intent_data.json"
        with open(lang_file, "w", encoding="utf-8") as f:
            json.dump(lang_data, f, ensure_ascii=False, indent=2)
        logger.info(f"  {lang_folder}: {len(lang_data)} samples → {lang_file}")

    return True


# ═══════════════════════════════════════════════════════════════
#  DATASET 4 — Weather Data (NASA POWER - Auto-fetch)
# ═══════════════════════════════════════════════════════════════
def download_weather_samples():
    """
    Pre-fetch weather samples for key Indian regions from NASA POWER.
    These are used to build the weather cache.
    """
    logger.info("=" * 55)
    logger.info("Pre-fetching Weather Samples (NASA POWER)")
    logger.info("=" * 55)

    WX_DIR.mkdir(parents=True, exist_ok=True)

    import json
    import datetime

    if not _REQUESTS:
        logger.error("requests not installed")
        return False

    REGIONS = {
        "tamil_nadu": (11.1271, 78.6569),
        "punjab": (30.9010, 75.8573),
        "andhra_pradesh": (15.9129, 79.7400),
        "maharashtra": (19.7515, 75.7139),
    }

    end_date = datetime.date.today() - datetime.timedelta(days=2)  # NASA has 2-day lag
    start_date = end_date - datetime.timedelta(days=30)

    PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M"
    saved = 0

    for region_name, (lat, lon) in REGIONS.items():
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters={PARAMS}&community=AG"
            f"&longitude={lon}&latitude={lat}"
            f"&start={start_date.strftime('%Y%m%d')}"
            f"&end={end_date.strftime('%Y%m%d')}&format=JSON"
        )
        try:
            logger.info(f"  Fetching {region_name}...")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            out_file = WX_DIR / f"{region_name}_weather.json"
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"  Saved: {out_file}")
            saved += 1
        except Exception as e:
            logger.warning(f"  Failed {region_name}: {e}")

    logger.info(f"Weather samples: {saved}/{len(REGIONS)} regions saved")
    return saved > 0


# ═══════════════════════════════════════════════════════════════
#  HELPER: Create all empty placeholder directories
# ═══════════════════════════════════════════════════════════════
def create_directory_structure():
    """Create all required directories with README placeholders."""
    dirs = [
        RAW_DIR / "images" / "plantvillage",
        RAW_DIR / "images" / "rice_leaf",
        RAW_DIR / "nlp" / "english",
        RAW_DIR / "nlp" / "tamil",
        RAW_DIR / "nlp" / "telugu",
        RAW_DIR / "nlp" / "hindi",
        RAW_DIR / "weather" / "nasa_power",
        RAW_DIR / "kg_sources",
        BASE_DIR / "processed" / "multilingual_tokens",
        BASE_DIR / "processed" / "image_tensors",
        BASE_DIR / "processed" / "weather_vectors",
        BASE_DIR / "processed" / "kg_triples",
        BASE_DIR / "embeddings" / "document_embeddings",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        placeholder = d / ".gitkeep"
        if not placeholder.exists():
            placeholder.touch()

    logger.info(f"Created {len(dirs)} directories")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AgriMultimodalQA datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--plantvillage", action="store_true", help="Download PlantVillage")
    parser.add_argument("--rice", action="store_true", help="Download Rice Leaf Disease")
    parser.add_argument("--nlp", action="store_true", help="Create NLP seed data")
    parser.add_argument("--weather", action="store_true", help="Pre-fetch weather samples")
    parser.add_argument("--structure", action="store_true", help="Create directory structure only")
    args = parser.parse_args()

    # Always create structure first
    create_directory_structure()

    if args.all or args.plantvillage:
        download_plantvillage()

    if args.all or args.rice:
        download_rice_leaf()

    if args.all or args.nlp:
        create_nlp_dataset()

    if args.all or args.weather:
        download_weather_samples()

    if args.structure and not any([args.all, args.plantvillage, args.rice, args.nlp, args.weather]):
        logger.info("Directory structure created. Run with --all to download datasets.")

    if not any(vars(args).values()):
        parser.print_help()
        print("\n Quick start:")
        print("   python data/download_datasets.py --structure   # Create folders only")
        print("   python data/download_datasets.py --nlp         # Create NLP seed data (no download)")
        print("   python data/download_datasets.py --weather     # Fetch weather from NASA (free)")
        print("   python data/download_datasets.py --rice        # Download rice leaf dataset")
        print("   python data/download_datasets.py --all         # Download everything")
