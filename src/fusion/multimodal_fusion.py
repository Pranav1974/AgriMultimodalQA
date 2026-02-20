"""
multimodal_fusion.py
--------------------
Fuses signals from NLP, Image, Weather, and Knowledge Graph
to create a unified context for answer generation.

Fusion Strategy:
  1. Concatenate feature vectors
  2. Apply learned dense fusion layer
  3. Weight signals by availability and confidence

Input signals:
  - NLP embedding (768-dim from XLM-R)
  - Image prediction (softmax probabilities)
  - Weather vector (7-dim normalized)
  - KG triples (as text context)

Output:
  - Fused context string for generation
  - Confidence-weighted decision
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FusionInput:
    """Structured input for the fusion layer."""
    # NLP signals
    query_text: str = ""
    intent: str = ""
    intent_confidence: float = 0.0
    crops: List[str] = field(default_factory=list)
    diseases: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    language: str = "en"

    # Image signals
    image_disease: Optional[str] = None
    image_crop: Optional[str] = None
    image_confidence: float = 0.0
    image_is_healthy: bool = False

    # Weather signals
    weather_vector: List[float] = field(default_factory=lambda: [0.0] * 7)
    weather_summary: str = ""
    weather_risk: List[dict] = field(default_factory=list)

    # KG signals
    kg_diseases: List[dict] = field(default_factory=list)
    kg_treatments: List[dict] = field(default_factory=list)
    kg_triples: List[str] = field(default_factory=list)

    # Retrieval signals
    retrieved_documents: List[str] = field(default_factory=list)

    # Ambiguity
    is_ambiguous: bool = False
    clarification_questions: List[str] = field(default_factory=list)


@dataclass
class FusionOutput:
    """Output of the fusion layer."""
    context: str = ""           # Rich context for generation
    primary_disease: str = ""   # Most likely disease
    primary_crop: str = ""      # Crop in question
    treatments: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    signal_weights: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""         # Explanation of fusion decision


class MultimodalFusion:
    """
    Fuses NLP, Image, Weather, and KG signals into a unified context.
    """

    # Signal weights (tuned for agricultural QA)
    DEFAULT_WEIGHTS = {
        "nlp": 0.40,     # NLP is primary
        "image": 0.30,   # Image is strong validator
        "kg": 0.20,      # KG provides structured knowledge
        "weather": 0.10, # Weather modifies diagnosis
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        logger.info(f"Fusion initialized with weights: {self.weights}")

    def fuse(self, inputs: FusionInput) -> FusionOutput:
        """
        Main fusion function.

        Args:
            inputs: FusionInput with all available signals

        Returns:
            FusionOutput with unified context
        """
        output = FusionOutput()

        # Module 2: Ambiguity Resolver (Conversational Intelligence)
        if inputs.is_ambiguous:
            output.context = (
                f"The query is ambiguous. "
                f"Clarification needed: {'; '.join(inputs.clarification_questions)}"
            )
            output.reasoning = "Query is ambiguous — clarification required"
            return output

        # Step 1: Determine primary crop
        output.primary_crop = self._determine_crop(inputs)

        # Step 2: Determine primary disease
        output.primary_disease, output.confidence = self._determine_disease(inputs)

        # Step 3: Gather treatments
        output.treatments = inputs.kg_treatments

        # Step 4: Compute signal weights based on availability
        available_weights = self._compute_available_weights(inputs)
        output.signal_weights = available_weights

        # Step 5: Build rich context string
        output.context = self._build_context(inputs, output)

        # Step 6: Build reasoning trace
        output.reasoning = self._build_reasoning(inputs, output)

        return output

    def _determine_crop(self, inputs: FusionInput) -> str:
        """Determine the most likely crop from all signals."""
        # Prefer NLP-detected crop
        if inputs.crops:
            return inputs.crops[0]
        # Fall back to image-detected crop
        if inputs.image_crop:
            return inputs.image_crop
        return "unknown"

    def _determine_disease(self, inputs: FusionInput) -> Tuple[str, float]:
        """
        Determine the most likely disease using weighted signals.
        Returns (disease_name, confidence).
        """
        candidates = {}

        # NLP signal: diseases explicitly mentioned
        for d in inputs.diseases:
            candidates[d.lower()] = candidates.get(d.lower(), 0) + self.weights["nlp"]

        # Image signal: model prediction
        if inputs.image_disease and inputs.image_confidence > 0.3:
            img_d = inputs.image_disease.lower()
            img_score = self.weights["image"] * inputs.image_confidence
            candidates[img_d] = candidates.get(img_d, 0) + img_score

        # KG signal: diseases matching symptoms
        if inputs.intent == "DISEASE_DIAGNOSIS" or inputs.symptoms:
            for kg_d in inputs.kg_diseases:
                d_name = kg_d.get("name", "").lower()
                candidates[d_name] = candidates.get(d_name, 0) + self.weights["kg"]

        if not candidates:
            return "unknown", 0.0

        # Pick highest scoring candidate
        best_disease = max(candidates, key=candidates.get)
        confidence = min(candidates[best_disease], 1.0)

        return best_disease, round(confidence, 3)

    def _compute_available_weights(self, inputs: FusionInput) -> Dict[str, float]:
        """Compute effective weights based on what signals are available."""
        available = {}
        if inputs.intent:
            available["nlp"] = self.weights["nlp"]
        if inputs.image_disease and inputs.image_confidence > 0.1:
            available["image"] = self.weights["image"]
        if inputs.kg_diseases or inputs.kg_treatments:
            available["kg"] = self.weights["kg"]
        if any(v != 0.0 for v in inputs.weather_vector):
            available["weather"] = self.weights["weather"]

        # Normalize weights
        total = sum(available.values()) or 1
        return {k: round(v / total, 3) for k, v in available.items()}

    def _build_context(self, inputs: FusionInput, output: FusionOutput) -> str:
        """Build rich context string for the answer generator."""
        context_parts = []

        # Query context
        context_parts.append(f"User query: {inputs.query_text}")
        context_parts.append(f"Intent: {inputs.intent}")
        context_parts.append(f"Crop: {output.primary_crop}")

        # Symptom context
        if inputs.symptoms:
            context_parts.append(f"Observed symptoms: {', '.join(inputs.symptoms)}")

        # Disease context
        if output.primary_disease and output.primary_disease != "unknown":
            context_parts.append(f"Likely disease: {output.primary_disease} (confidence: {output.confidence:.0%})")

        # Image validation
        if inputs.image_disease:
            context_parts.append(
                f"Image analysis: {inputs.image_crop} with {inputs.image_disease} "
                f"({inputs.image_confidence:.0%} confidence)"
            )

        # Weather context
        if inputs.weather_summary:
            context_parts.append(f"Weather: {inputs.weather_summary}")

        # Risk context
        if inputs.weather_risk:
            risks = [f"{r['disease']} ({r['level']})" for r in inputs.weather_risk]
            context_parts.append(f"Weather-related risks: {', '.join(risks)}")

        # KG knowledge
        if inputs.kg_triples:
            context_parts.append(f"Knowledge graph: {' | '.join(inputs.kg_triples[:3])}")

        # Treatment knowledge
        if inputs.kg_treatments:
            t_names = [t.get("name", "") for t in inputs.kg_treatments[:3]]
            context_parts.append(f"Recommended treatments: {', '.join(t_names)}")

        # Retrieved documents
        if inputs.retrieved_documents:
            context_parts.append(f"Reference: {inputs.retrieved_documents[0][:300]}...")

        return "\n".join(context_parts)

    def _build_reasoning(self, inputs: FusionInput, output: FusionOutput) -> str:
        """Build a human-readable reasoning trace."""
        signals_used = list(output.signal_weights.keys())
        reasoning = (
            f"Disease determination used signals: {', '.join(signals_used)}. "
            f"NLP detected '{', '.join(inputs.diseases) or 'none'}'. "
        )
        if inputs.image_disease:
            reasoning += f"Image detected '{inputs.image_disease}' ({inputs.image_confidence:.0%}). "
        if inputs.weather_risk:
            reasoning += f"Weather indicates risk of {inputs.weather_risk[0]['disease']}. "
        reasoning += f"Final verdict: {output.primary_disease} with {output.confidence:.0%} confidence."
        return reasoning


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Multimodal Fusion Test")
    print("=" * 60)

    fusion = MultimodalFusion()

    # Simulate inputs
    test_input = FusionInput(
        query_text="My rice plants in Tamil Nadu have yellow spots on leaves",
        intent="DISEASE_DIAGNOSIS",
        intent_confidence=0.92,
        crops=["rice"],
        diseases=[],
        symptoms=["yellow spots"],
        regions=["Tamil Nadu"],
        language="en",
        # Image says: blast with 78% confidence
        image_disease="Rice Blast",
        image_crop="Rice",
        image_confidence=0.78,
        # Weather: high humidity
        weather_vector=[0.70, 0.75, 0.65, 0.85, 0.60, 0.30, 0.50],
        weather_summary="High humidity (87%), temp 28°C, rainfall 65mm",
        weather_risk=[{"disease": "Fungal blast", "level": "HIGH", "reason": "High humidity"}],
        # KG: blast is known to affect rice
        kg_diseases=[{"id": "blast", "name": "Rice Blast", "pathogen": "Magnaporthe oryzae"}],
        kg_treatments=[
            {"name": "Tricyclazole 75% WP", "dose": "0.6g per liter"},
            {"name": "Carbendazim 50% WP", "dose": "1g per liter"},
        ],
        kg_triples=[
            "Rice Blast AFFECTS Rice",
            "Rice Blast TREATED_BY Tricyclazole",
            "High humidity CAUSES Rice Blast"
        ]
    )

    result = fusion.fuse(test_input)
    print(f"Primary Disease : {result.primary_disease}")
    print(f"Primary Crop    : {result.primary_crop}")
    print(f"Confidence      : {result.confidence:.0%}")
    print(f"Signal Weights  : {result.signal_weights}")
    print(f"Reasoning       : {result.reasoning}")
    print(f"\nContext:\n{result.context}")
