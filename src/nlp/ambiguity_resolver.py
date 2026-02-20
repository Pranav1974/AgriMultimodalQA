"""
ambiguity_resolver.py
---------------------
Resolves ambiguous agricultural queries by detecting missing information
and generating clarifying questions.

AMBIGUITY TYPES:
  1. MISSING_CROP      : No crop mentioned → "Which crop are you asking about?"
  2. MISSING_SYMPTOM   : Disease/concern without symptom details
  3. MISSING_REGION    : Weather-dependent query without location
  4. VAGUE_SYMPTOM     : Too general (e.g., "bad leaves" → ask for color/pattern)
  5. MISSING_TIME      : Seasonal query without timeframe
  6. COMPOUND_QUERY    : Multiple different questions in one

RESOLUTION STRATEGY:
  - Rule-based ambiguity scoring
  - Targeted clarification question generation
  - Confidence threshold to avoid over-questioning
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


# ── Ambiguity Configuration ──────────────────────────────────

AMBIGUITY_CONFIDENCE_THRESHOLD = 0.5  # Only ask if confidence < this

CLARIFICATION_TEMPLATES = {
    "MISSING_CROP": [
        "Which crop are you asking about? (e.g., rice, wheat, cotton)",
        "Could you specify the crop name?",
        "What crop is affected?",
    ],
    "MISSING_SYMPTOM": [
        "What symptoms are you observing on your plant?",
        "Can you describe what the affected area looks like? (color, pattern, size)",
        "Are the leaves showing yellow, brown, or white spots?",
    ],
    "VAGUE_SYMPTOM": [
        "Can you describe the symptom in more detail?",
        "What color are the spots or affected areas?",
        "Are the spots on leaves, stem, or roots?",
    ],
    "MISSING_REGION": [
        "Which region or state are you located in?",
        "Could you share your location for location-specific advice?",
        "What is your farming region? (e.g., Punjab, Tamil Nadu)",
    ],
    "MISSING_TIME": [
        "Which growing season is this? (Kharif/Rabi/Summer)",
        "What time of year is it in your area?",
        "How old are your plants currently?",
    ],
    "COMPOUND_QUERY": [
        "I'll address your questions one by one. Which is most urgent?",
        "Could you focus on one question at a time?",
    ],
}

# Vague terms that trigger VAGUE_SYMPTOM
VAGUE_TERMS = {
    "bad", "wrong", "problem", "issue", "sick", "dying",
    "not good", "unhealthy", "trouble", "damage", "affected"
}


@dataclass
class AmbiguityResult:
    """Result of ambiguity analysis."""
    is_ambiguous: bool
    ambiguity_types: List[str] = field(default_factory=list)
    clarification_questions: List[str] = field(default_factory=list)
    severity: float = 0.0  # 0.0 = clear, 1.0 = very ambiguous

    def to_dict(self) -> dict:
        return {
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_types": self.ambiguity_types,
            "clarification_questions": self.clarification_questions,
            "severity": round(self.severity, 2),
            "primary_question": self.clarification_questions[0] if self.clarification_questions else None,
        }


@dataclass
class QueryContext:
    """Extracted context from NER and intent classification."""
    text: str
    intent: str
    crops: List[str] = field(default_factory=list)
    diseases: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    chemicals: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    quantities: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)


def _check_missing_crop(context: QueryContext) -> Tuple[bool, float]:
    """Check if crop is missing in a crop-specific query."""
    crop_required_intents = {
        "DISEASE_DIAGNOSIS", "TREATMENT_ADVICE",
        "FERTILIZER_QUERY", "PEST_CONTROL",
        "SOIL_QUERY", "IRRIGATION_QUERY"
    }
    if context.intent in crop_required_intents and not context.crops:
        return True, 0.8
    return False, 0.0


def _check_vague_symptom(context: QueryContext) -> Tuple[bool, float]:
    """Check if symptom description is too vague."""
    text_lower = context.text.lower()
    if context.intent == "DISEASE_DIAGNOSIS":
        # No specific symptom detected
        if not context.symptoms and not context.diseases:
            return True, 0.7
        # Vague terms used
        vague_count = sum(1 for vt in VAGUE_TERMS if vt in text_lower)
        if vague_count > 0 and not context.symptoms:
            return True, 0.6
    return False, 0.0


def _check_missing_region(context: QueryContext) -> Tuple[bool, float]:
    """Check if region is missing for weather-sensitive queries."""
    if context.intent in {"WEATHER_ADVICE", "CROP_RECOMMENDATION"} and not context.regions:
        return True, 0.6
    return False, 0.0


def _check_missing_time(context: QueryContext) -> Tuple[bool, float]:
    """Check if time context is missing for seasonal queries."""
    if context.intent in {"CROP_RECOMMENDATION", "WEATHER_ADVICE"} and not context.times:
        return True, 0.4
    return False, 0.0


def _check_compound_query(context: QueryContext) -> Tuple[bool, float]:
    """Check if multiple intents are present in one query."""
    text_lower = context.text.lower()
    question_count = text_lower.count("?") + text_lower.count(" and ") + text_lower.count(" also ")
    if question_count >= 2:
        return True, 0.5
    return False, 0.0


class AmbiguityResolver:
    """
    Detects ambiguities in agricultural queries and generates
    targeted clarification questions.
    """

    def __init__(self, threshold: float = AMBIGUITY_CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        logger.info(f"AmbiguityResolver initialized (threshold={threshold})")

    def analyze(
        self,
        text: str,
        intent: str,
        ner_result: dict,
        intent_confidence: float = 1.0
    ) -> AmbiguityResult:
        """
        Analyze a query for ambiguity.

        Args:
            text (str): The query text (in English)
            intent (str): Classified intent
            ner_result (dict): Output from NER model
            intent_confidence (float): How confident the intent classification was

        Returns:
            AmbiguityResult
        """
        context = QueryContext(
            text=text,
            intent=intent,
            crops=ner_result.get("crops", []),
            diseases=ner_result.get("diseases", []),
            symptoms=ner_result.get("symptoms", []),
            chemicals=ner_result.get("chemicals", []),
            regions=ner_result.get("regions", []),
            quantities=ner_result.get("quantities", []),
            times=ner_result.get("times", []),
        )

        # If intent itself is unclear, that's a top-level ambiguity
        if intent_confidence < self.threshold:
            return AmbiguityResult(
                is_ambiguous=True,
                ambiguity_types=["UNCLEAR_INTENT"],
                clarification_questions=[
                    "Could you clarify what you'd like to know?",
                    "Are you asking about a plant disease, fertilizer, or something else?"
                ],
                severity=1.0 - intent_confidence
            )

        # Run ambiguity checks
        checks = [
            ("MISSING_CROP", _check_missing_crop(context)),
            ("VAGUE_SYMPTOM", _check_vague_symptom(context)),
        ]

        ambiguity_types = []
        questions = []
        total_severity = 0.0

        for amb_type, (is_amb, severity) in checks:
            if is_amb:
                ambiguity_types.append(amb_type)
                total_severity += severity
                templates = CLARIFICATION_TEMPLATES.get(amb_type, [])
                if templates:
                    questions.append(templates[0])  # Pick first template

        if not ambiguity_types:
            return AmbiguityResult(is_ambiguous=False, severity=0.0)

        avg_severity = total_severity / len(checks)
        return AmbiguityResult(
            is_ambiguous=True,
            ambiguity_types=ambiguity_types,
            clarification_questions=questions[:2],  # Max 2 questions at a time
            severity=min(avg_severity, 1.0)
        )

    def resolve_with_context(
        self,
        original_query: str,
        clarification_answers: Dict[str, str],
        intent: str,
        ner_result: dict
    ) -> dict:
        """
        Merge original query with user-provided clarifications.

        Args:
            original_query: Original ambiguous query
            clarification_answers: {"MISSING_CROP": "rice", "MISSING_REGION": "Punjab"}
            intent: Original intent
            ner_result: Original NER result

        Returns:
            dict: Updated context with resolved entities
        """
        updated_ner = dict(ner_result)

        for amb_type, answer in clarification_answers.items():
            answer = answer.strip()
            if amb_type == "MISSING_CROP":
                updated_ner.setdefault("crops", []).append(answer)
            elif amb_type in ("MISSING_SYMPTOM", "VAGUE_SYMPTOM"):
                updated_ner.setdefault("symptoms", []).append(answer)
            elif amb_type == "MISSING_REGION":
                updated_ner.setdefault("regions", []).append(answer)
            elif amb_type == "MISSING_TIME":
                updated_ner.setdefault("times", []).append(answer)

        # Construct enriched query
        enriched = f"{original_query}. Crop: {updated_ner.get('crops', [])}. Symptoms: {updated_ner.get('symptoms', [])}."

        return {
            "enriched_query": enriched,
            "updated_ner": updated_ner,
            "intent": intent,
            "resolved": True
        }


# Singleton
_resolver_instance: Optional[AmbiguityResolver] = None


def get_ambiguity_resolver() -> AmbiguityResolver:
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = AmbiguityResolver()
    return _resolver_instance


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    resolver = AmbiguityResolver()

    test_cases = [
        {
            "text": "My plants are having a bad problem",
            "intent": "DISEASE_DIAGNOSIS",
            "ner_result": {"crops": [], "symptoms": [], "diseases": [], "regions": [], "times": []},
            "intent_confidence": 0.75
        },
        {
            "text": "What fertilizer for rice?",
            "intent": "FERTILIZER_QUERY",
            "ner_result": {"crops": ["rice"], "symptoms": [], "diseases": [], "regions": [], "times": []},
            "intent_confidence": 0.90
        },
        {
            "text": "Is weather good for sowing?",
            "intent": "WEATHER_ADVICE",
            "ner_result": {"crops": [], "symptoms": [], "diseases": [], "regions": [], "times": []},
            "intent_confidence": 0.82
        },
        {
            "text": "My rice plants in Tamil Nadu have yellow spots",
            "intent": "DISEASE_DIAGNOSIS",
            "ner_result": {
                "crops": ["rice"], "symptoms": ["yellow", "spots"],
                "diseases": [], "regions": ["Tamil Nadu"], "times": []
            },
            "intent_confidence": 0.95
        },
    ]

    print("=" * 60)
    print("Ambiguity Resolution Test")
    print("=" * 60)
    for tc in test_cases:
        result = resolver.analyze(
            tc["text"], tc["intent"], tc["ner_result"], tc["intent_confidence"]
        )
        r = result.to_dict()
        print(f"Query      : {tc['text']}")
        print(f"Ambiguous  : {r['is_ambiguous']} (severity: {r['severity']})")
        print(f"Types      : {r['ambiguity_types']}")
        print(f"Question   : {r['primary_question']}")
        print("-" * 40)
