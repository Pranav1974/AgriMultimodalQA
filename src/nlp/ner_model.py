"""
ner_model.py
-----------
Named Entity Recognition for agricultural queries.

ENTITY TYPES:
  - CROP       : rice, wheat, paddy, cotton, sugarcane
  - DISEASE    : blast, blight, rust, rot, mosaic
  - SYMPTOM    : yellow spots, wilting, leaf curl, browning
  - CHEMICAL   : urea, DAP, carbendazim, chlorpyrifos
  - REGION     : Punjab, Tamil Nadu, Andhra Pradesh, India
  - QUANTITY   : 50 kg, 2 liters, per acre
  - TIME       : kharif, rabi, summer, monsoon, June

Architecture:
  - XLM-RoBERTa-base with Token Classification head
  - Fine-tuned on agricultural NER dataset
  - Rule-based dictionary fallback for known entities
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using rule-based NER only.")

# ── Entity Labels (BIO tagging) ──────────────────────────────
NER_LABELS = [
    "O",
    "B-CROP", "I-CROP",
    "B-DISEASE", "I-DISEASE",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-CHEMICAL", "I-CHEMICAL",
    "B-REGION", "I-REGION",
    "B-QUANTITY", "I-QUANTITY",
    "B-TIME", "I-TIME",
]

id2tag = {i: label for i, label in enumerate(NER_LABELS)}
tag2id = {label: i for i, label in enumerate(NER_LABELS)}

# ── Agricultural knowledge dictionaries ──────────────────────
CROP_DICTIONARY = {
    "rice", "paddy", "wheat", "corn", "maize", "cotton",
    "sugarcane", "soybean", "groundnut", "potato", "tomato",
    "onion", "chili", "chile", "turmeric", "ginger", "banana",
    # Tamil
    "நெல்", "கோதுமை", "கரும்பு", "பருத்தி",
    # Hindi
    "धान", "गेहूं", "गन्ना", "कपास",
    # Telugu
    "వరి", "గోధుమ", "చెరకు",
}

DISEASE_DICTIONARY = {
    "blast", "blight", "rust", "rot", "mosaic", "smut",
    "wilt", "mildew", "anthracnose", "scorch", "yellowing",
    "leaf spot", "neck rot", "brown spot", "sheath blight",
    "sheath rot", "false smut", "bacterial leaf blight",
    "rice tungro", "downy mildew", "powdery mildew",
    "crown rot", "foot rot", "stem rot", "root rot",
}

SYMPTOM_DICTIONARY = {
    "yellow", "yellowing", "brown", "browning", "white",
    "spots", "lesions", "wilting", "wilt", "curling",
    "drooping", "streaks", "blotches", "necrosis", "chlorosis",
    "stunted growth", "pale", "discolored", "drying", "defoliation",
}

CHEMICAL_DICTIONARY = {
    "urea", "dap", "mop", "npk", "carbendazim", "mancozeb",
    "chlorpyrifos", "imidacloprid", "tricyclazole", "propiconazole",
    "copper oxychloride", "bordeaux mixture", "neem oil",
    "azadirachtin", "glyphosate", "2,4-d", "atrazine",
}

REGION_DICTIONARY = {
    "punjab", "haryana", "uttar pradesh", "bihar", "west bengal",
    "andhra pradesh", "telangana", "karnataka", "kerala", "tamil nadu",
    "maharashtra", "gujarat", "rajasthan", "madhya pradesh",
    "india", "pakistan", "bangladesh",
}

TIME_DICTIONARY = {
    "kharif", "rabi", "summer", "monsoon", "winter",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "spring", "autumn", "fall", "pre-monsoon", "post-monsoon",
}


def rule_based_ner(text: str) -> List[Dict]:
    """
    Dictionary-based NER using known agricultural terms.

    Returns:
        List of entities: [{"text": str, "label": str, "start": int, "end": int}]
    """
    text_lower = text.lower()
    entities = []

    def find_entities(dictionary, label, text_lower, text_original):
        found = []
        for term in sorted(dictionary, key=len, reverse=True):  # Longest match first
            term_lower = term.lower()
            start = 0
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                # Check word boundaries
                before = text_lower[pos - 1] if pos > 0 else ' '
                after = text_lower[pos + len(term_lower)] if pos + len(term_lower) < len(text_lower) else ' '
                if not before.isalnum() and not after.isalnum():
                    found.append({
                        "text": text_original[pos:pos + len(term_lower)],
                        "label": label,
                        "start": pos,
                        "end": pos + len(term_lower)
                    })
                start = pos + 1
        return found

    entities.extend(find_entities(CROP_DICTIONARY, "CROP", text_lower, text))
    entities.extend(find_entities(DISEASE_DICTIONARY, "DISEASE", text_lower, text))
    entities.extend(find_entities(SYMPTOM_DICTIONARY, "SYMPTOM", text_lower, text))
    entities.extend(find_entities(CHEMICAL_DICTIONARY, "CHEMICAL", text_lower, text))
    entities.extend(find_entities(REGION_DICTIONARY, "REGION", text_lower, text))
    entities.extend(find_entities(TIME_DICTIONARY, "TIME", text_lower, text))

    # Detect quantities (e.g., "50 kg", "2 liters per acre")
    quantity_pattern = r'\d+\.?\d*\s*(?:kg|g|ml|l|litre|liter|ton|tonne|acre|hectare|per\s+acre|per\s+hectare)'
    for match in re.finditer(quantity_pattern, text_lower):
        entities.append({
            "text": text[match.start():match.end()],
            "label": "QUANTITY",
            "start": match.start(),
            "end": match.end()
        })

    # Sort by start position & remove overlaps
    entities.sort(key=lambda x: x["start"])
    filtered = []
    last_end = -1
    for ent in entities:
        if ent["start"] >= last_end:
            filtered.append(ent)
            last_end = ent["end"]

    return filtered


class NERModel:
    """
    Agricultural NER model using XLM-RoBERTa with rule-based fallback.
    """

    MODEL_NAME = "xlm-roberta-base"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self.ner_pipeline = None

        if _TORCH_AVAILABLE:
            load_path = model_path or self.MODEL_NAME
            try:
                logger.info(f"Loading NER pipeline from {load_path}")
                tokenizer = AutoTokenizer.from_pretrained(load_path)

                if model_path:  # Only if fine-tuned model exists
                    model = AutoModelForTokenClassification.from_pretrained(
                        load_path,
                        num_labels=len(NER_LABELS),
                        id2label=id2tag,
                        label2id=tag2id,
                        ignore_mismatched_sizes=True
                    )
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="simple",
                        device=0 if self.device == "cuda" else -1
                    )
                    logger.info("NER model loaded")
                else:
                    logger.info("No fine-tuned model, using rule-based NER")
            except Exception as e:
                logger.error(f"NER model load failed: {e}")
                self.ner_pipeline = None

    def predict(self, text: str) -> dict:
        """
        Extract named entities from text.

        Returns:
            dict: {
                "entities": list,
                "method": str,
                "crops": list,
                "diseases": list,
                "symptoms": list,
                "chemicals": list,
                "regions": list,
                "quantities": list,
                "times": list
            }
        """
        if self.ner_pipeline:
            try:
                return self._predict_with_model(text)
            except Exception as e:
                logger.warning(f"NER model failed: {e}")

        return self._predict_rule_based(text)

    def _predict_with_model(self, text: str) -> dict:
        """Use fine-tuned model for NER."""
        raw_entities = self.ner_pipeline(text)
        entities = []
        for ent in raw_entities:
            entities.append({
                "text": ent["word"],
                "label": ent["entity_group"].replace("B-", "").replace("I-", ""),
                "start": ent["start"],
                "end": ent["end"],
                "score": round(ent["score"], 4)
            })
        return self._structure_entities(entities, "xlm_roberta")

    def _predict_rule_based(self, text: str) -> dict:
        """Use rule-based NER."""
        entities = rule_based_ner(text)
        return self._structure_entities(entities, "rule_based")

    def _structure_entities(self, entities: list, method: str) -> dict:
        """Structure entities by type."""
        result = {
            "entities": entities,
            "method": method,
            "crops": [],
            "diseases": [],
            "symptoms": [],
            "chemicals": [],
            "regions": [],
            "quantities": [],
            "times": [],
        }
        for ent in entities:
            label = ent["label"].upper()
            if label == "CROP":
                result["crops"].append(ent["text"])
            elif label == "DISEASE":
                result["diseases"].append(ent["text"])
            elif label == "SYMPTOM":
                result["symptoms"].append(ent["text"])
            elif label == "CHEMICAL":
                result["chemicals"].append(ent["text"])
            elif label == "REGION":
                result["regions"].append(ent["text"])
            elif label == "QUANTITY":
                result["quantities"].append(ent["text"])
            elif label == "TIME":
                result["times"].append(ent["text"])
        return result


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "My rice plants in Tamil Nadu show yellow spots and wilting",
        "Apply carbendazim 50g per acre for wheat blast treatment",
        "Cotton in Punjab has aphid infestation during kharif season",
        "Paddy leaves have brown spots near roots",
    ]
    print("=" * 60)
    print("NER Test (Rule-Based)")
    print("=" * 60)
    ner = NERModel(model_path=None)
    for query in test_queries:
        result = ner.predict(query)
        print(f"Query    : {query}")
        print(f"Crops    : {result['crops']}")
        print(f"Diseases : {result['diseases']}")
        print(f"Symptoms : {result['symptoms']}")
        print(f"Chemicals: {result['chemicals']}")
        print(f"Regions  : {result['regions']}")
        print("-" * 40)
