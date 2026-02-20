"""
intent_classifier.py
--------------------
Classifies the intent of an agricultural query into one of:

INTENT CLASSES:
  - DISEASE_DIAGNOSIS   : "What disease does my plant have?"
  - TREATMENT_ADVICE    : "How to treat yellow leaf spot?"
  - FERTILIZER_QUERY    : "What fertilizer should I use?"
  - WEATHER_ADVICE      : "Is today's weather good for sowing?"
  - PEST_CONTROL        : "How to control aphids?"
  - SOIL_QUERY          : "What soil type is needed for paddy?"
  - IRRIGATION_QUERY    : "How often should I water rice?"
  - CROP_RECOMMENDATION : "What crop should I grow now?"
  - GENERAL_AGRI_INFO   : "Tell me about paddy cultivation"
  - OUT_OF_SCOPE        : Non-agricultural queries

Architecture:
  - XLM-RoBERTa base (multilingual)
  - Fine-tuned on agricultural intent dataset
  - Rule-based fallback for high-confidence keywords
"""

import os
import json
from typing import Optional, List, Dict, Tuple
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch/transformers not available")

# ── Intent Labels ────────────────────────────────────────────
INTENT_LABELS = [
    "DISEASE_DIAGNOSIS",
    "TREATMENT_ADVICE",
    "FERTILIZER_QUERY",
    "WEATHER_ADVICE",
    "PEST_CONTROL",
    "SOIL_QUERY",
    "IRRIGATION_QUERY",
    "CROP_RECOMMENDATION",
    "GENERAL_AGRI_INFO",
    "OUT_OF_SCOPE",
]

NUM_INTENTS = len(INTENT_LABELS)
id2label = {i: label for i, label in enumerate(INTENT_LABELS)}
label2id = {label: i for i, label in enumerate(INTENT_LABELS)}

# ── Rule-based keyword fallback ──────────────────────────────
INTENT_KEYWORDS = {
    "DISEASE_DIAGNOSIS": [
        "disease", "infection", "spots", "blight", "rot", "rust",
        "fungal", "bacterial", "viral", "symptom", "affected",
        "yellow", "brown", "white", "lesion", "wilt", "dying"
    ],
    "TREATMENT_ADVICE": [
        "treat", "treatment", "cure", "control", "spray", "apply",
        "medicine", "fungicide", "bactericide", "how to fix", "solution"
    ],
    "FERTILIZER_QUERY": [
        "fertilizer", "fertiliser", "npk", "nitrogen", "phosphorus",
        "potassium", "urea", "compost", "manure", "nutrient", "deficiency"
    ],
    "WEATHER_ADVICE": [
        "weather", "rain", "temperature", "humidity", "climate",
        "monsoon", "season", "forecast", "sowing time", "harvest time"
    ],
    "PEST_CONTROL": [
        "pest", "insect", "bug", "aphid", "whitefly", "mite",
        "caterpillar", "borer", "grasshopper", "locust", "pesticide"
    ],
    "SOIL_QUERY": [
        "soil", "ph", "acidic", "alkaline", "clay", "loam",
        "sandy", "organic matter", "soil test", "drainage"
    ],
    "IRRIGATION_QUERY": [
        "water", "irrigation", "watering", "drip", "sprinkler",
        "flood", "moisture", "drought", "how often water"
    ],
    "CROP_RECOMMENDATION": [
        "which crop", "what crop", "suitable crop", "grow",
        "plant", "cultivate", "recommend", "best crop"
    ],
    "GENERAL_AGRI_INFO": [
        "how to", "what is", "tell me", "explain", "information",
        "cultivation", "farming", "agriculture", "method"
    ],
}


def rule_based_intent(text: str) -> Tuple[Optional[str], float]:
    """
    Simple keyword-based intent detection.
    Returns (intent, confidence) or (None, 0.0) if unclear.
    """
    text_lower = text.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return None, 0.0

    top_intent = max(scores, key=scores.get)
    # Normalize score (crude confidence) - base 0.5 + 0.3 per keyword match
    confidence = min(0.5 + (scores[top_intent] * 0.3), 1.0)
    return top_intent, confidence


class IntentClassifier:
    """
    XLM-RoBERTa based intent classifier with rule-based fallback.
    """

    MODEL_NAME = "xlm-roberta-base"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Path to fine-tuned model (None = use pretrained only)
            device: "cuda" / "cpu" / None (auto-detect)
        """
        if not _TORCH_AVAILABLE:
            self.model = None
            self.tokenizer = None
            logger.warning("Model not available, using rule-based fallback only")
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        load_path = model_path or self.MODEL_NAME

        try:
            logger.info(f"Loading tokenizer from {load_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)

            logger.info(f"Loading model from {load_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                load_path,
                num_labels=NUM_INTENTS,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True  # For fresh fine-tuning
            )
            self.model.eval()
            self.model.to(self.device)
            logger.info("Intent classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def predict(self, text: str) -> dict:
        """
        Predict intent of a query.

        Args:
            text (str): Input query (in English or any language)

        Returns:
            dict: {
                "intent": str,
                "confidence": float,
                "all_scores": dict,
                "method": str
            }
        """
        # Try deep learning model first
        if self.model is not None and self.tokenizer is not None:
            try:
                return self._predict_with_model(text)
            except Exception as e:
                logger.warning(f"Model prediction failed, using fallback: {e}")

        # Fall back to rule-based
        return self._predict_rule_based(text)

    def _predict_with_model(self, text: str) -> dict:
        """Use fine-tuned XLM-RoBERTa for prediction."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_id = probs.argmax().item()
        confidence = probs[pred_id].item()

        all_scores = {
            id2label[i]: round(probs[i].item(), 4)
            for i in range(NUM_INTENTS)
        }

        return {
            "intent": id2label[pred_id],
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "method": "xlm_roberta"
        }

    def _predict_rule_based(self, text: str) -> dict:
        """Rule-based fallback."""
        intent, confidence = rule_based_intent(text)
        if intent is None:
            intent = "GENERAL_AGRI_INFO"
            confidence = 0.3

        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": {},
            "method": "rule_based"
        }

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Predict intents for a batch of texts."""
        return [self.predict(t) for t in texts]


# Singleton
_classifier_instance: Optional[IntentClassifier] = None


def get_intent_classifier(model_path: Optional[str] = None) -> IntentClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier(model_path=model_path)
    return _classifier_instance


def classify_intent(text: str, model_path: Optional[str] = None) -> dict:
    return get_intent_classifier(model_path).predict(text)


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What disease does my rice plant have?",
        "How to treat yellow leaf blight in paddy?",
        "What fertilizer should I use for wheat?",
        "Is this weather suitable for sowing?",
        "How to control aphids in cotton?",
        "What soil type is best for sugarcane?",
        "How often should I water rice plants?",
        "Which crop should I grow in summer?",
        "Tell me about paddy cultivation methods",
        "What is the population of India?",  # Out of scope
    ]
    print("=" * 60)
    print("Intent Classification Test (Rule-Based Fallback)")
    print("=" * 60)
    classifier = IntentClassifier(model_path=None)
    for query in test_queries:
        result = classifier.predict(query)
        print(f"Query  : {query[:60]}")
        print(f"Intent : {result['intent']} ({result['confidence']:.2f}) [{result['method']}]")
        print("-" * 40)
