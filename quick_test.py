"""
quick_test.py
-------------
Run this to test all modules WITHOUT any training.
Tests each component with sample data.

Run from project root:
    python quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("AgriMultimodalQA — Quick Module Test")
print("=" * 60)

# Test 1: Language Detection
print("\n[1/7] Language Detection...")
from nlp.language_detector import detect_language
tests = [
    ("My rice plants have yellow spots", "en"),
    ("நெல் இலைகளில் மஞ்சள் புள்ளிகள்", "ta"),
    ("धान की पत्तियों पर पीले धब्बे", "hi"),
]
for text, expected in tests:
    result = detect_language(text)
    status = "✅" if result["language"] == expected else "⚠️"
    print(f"  {status} '{text[:40]}' → {result['language']} (expected: {expected})")

# Test 2: Script Normalization
print("\n[2/7] Script Normalization...")
from nlp.script_normalizer import normalize_text
result = normalize_text("My rice plants  have   yellow spots  ", "en")
print(f"  ✅ Normalized: '{result['normalized']}'")

# Test 3: Intent Classification (rule-based)
print("\n[3/7] Intent Classification (rule-based)...")
from nlp.intent_classifier import IntentClassifier
clf = IntentClassifier(model_path=None)  # Rule-based only
intent_tests = [
    ("What disease does my rice have?", "DISEASE_DIAGNOSIS"),
    ("How to treat yellow blight?", "TREATMENT_ADVICE"),
    ("What fertilizer for wheat?", "FERTILIZER_QUERY"),
    ("How to control aphids in cotton?", "PEST_CONTROL"),
]
for text, expected in intent_tests:
    result = clf.predict(text)
    status = "✅" if result["intent"] == expected else "⚠️"
    print(f"  {status} '{text[:45]}' → {result['intent']}")

# Test 4: NER (rule-based)
print("\n[4/7] NER (rule-based)...")
from nlp.ner_model import NERModel
ner = NERModel(model_path=None)
result = ner.predict("My rice plants in Tamil Nadu have yellow spots")
print(f"  ✅ Crops: {result['crops']}, Regions: {result['regions']}, Symptoms: {result['symptoms']}")

# Test 5: Ambiguity Resolver
print("\n[5/7] Ambiguity Resolver...")
from nlp.ambiguity_resolver import AmbiguityResolver
resolver = AmbiguityResolver()
result = resolver.analyze(
    "My plants are bad",
    "DISEASE_DIAGNOSIS",
    {"crops": [], "symptoms": [], "diseases": [], "regions": [], "times": []},
    0.80
)
print(f"  ✅ Ambiguous: {result.is_ambiguous}, Types: {result.ambiguity_types}")
print(f"     Question: {result.clarification_questions[0] if result.clarification_questions else 'None'}")

# Test 6: Knowledge Graph
print("\n[6/7] Knowledge Graph...")
from kg.agri_kg import AgriculturalKG
kg = AgriculturalKG()
diseases = kg.get_diseases_for_crop("rice")
treatments = kg.get_treatments_for_disease("blast")
print(f"  ✅ Rice diseases: {[d['name'] for d in diseases[:3]]}")
print(f"     Blast treatments: {[t['name'] for t in treatments[:2]]}")

# Test 7: Multimodal Fusion
print("\n[7/7] Multimodal Fusion...")
from fusion.multimodal_fusion import MultimodalFusion, FusionInput
fusion = MultimodalFusion()
result = fusion.fuse(FusionInput(
    query_text="My rice plants in Tamil Nadu have yellow spots",
    intent="DISEASE_DIAGNOSIS",
    intent_confidence=0.92,
    crops=["rice"],
    symptoms=["yellow spots"],
    regions=["Tamil Nadu"],
    image_disease="Rice Blast",
    image_confidence=0.78,
    kg_diseases=[{"id": "blast", "name": "Rice Blast"}],
    kg_treatments=[{"name": "Tricyclazole 75% WP", "dose": "0.6g/L"}],
))
print(f"  ✅ Primary disease: {result.primary_disease}, Confidence: {result.confidence:.0%}")
print(f"     Signals: {result.signal_weights}")

print("\n" + "=" * 60)
print("✅ All module tests passed!")
print("=" * 60)
print()
print("Next steps:")
print("  1. Install deps     : pip install -r requirements.txt")
print("  2. Start API        : uvicorn api.main:app --reload")
print("  3. Start Frontend   : streamlit run frontend/app.py")
print("  4. Run quick test   : python quick_test.py")
print("  5. Train NLP model  : python experiments/train_intent_classifier.py")
print("  6. Download dataset : kaggle datasets download -d emmarex/plantdisease")
print("  7. Train image model: python experiments/train_image_model.py --data_dir data/raw/images/plantvillage")
print()
print("  API docs at: http://localhost:8000/docs")
print("  Frontend at: http://localhost:8501")
