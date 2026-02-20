"""
pipeline.py
-----------
Main AgriMultimodalQA Pipeline.

Orchestrates all modules:
  1. Language Detection
  2. Script Normalization
  3. Translation (to English)
  4. Intent Classification
  5. NER (Crop, Disease, Symptom...)
  6. Ambiguity Resolution
  7. KG Query
  8. Weather Fetching
  9. Image Disease Classification (optional)
  10. Multimodal Fusion
  11. RAG Answer Generation

This is the single entry point for the system.
"""

import os
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

# ── Local imports ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.language_detector import detect_language
from nlp.script_normalizer import normalize_text
from nlp.translator import translate_to_english
from nlp.intent_classifier import IntentClassifier
from nlp.ner_model import NERModel
from nlp.ambiguity_resolver import AmbiguityResolver
from kg.agri_kg import AgriculturalKG
from weather.nasa_weather import get_weather_context, assess_disease_risk_from_weather
from fusion.multimodal_fusion import MultimodalFusion, FusionInput
from retrieval.rag_pipeline import AgriAnswerGenerator


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    intent_model_path: Optional[str] = None
    ner_model_path: Optional[str] = None
    image_model_path: Optional[str] = None
    generator_model_path: Optional[str] = None
    kg_path: Optional[str] = None
    enable_weather: bool = True
    enable_image: bool = True
    weather_days_back: int = 7
    top_k_docs: int = 3


@dataclass
class QueryResult:
    """Final result of the pipeline."""
    # Input
    original_query: str = ""
    language: str = "en"
    translated_query: str = ""

    # NLP
    intent: str = ""
    intent_confidence: float = 0.0
    crops: list = field(default_factory=list)
    diseases: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)
    regions: list = field(default_factory=list)

    # Ambiguity
    is_ambiguous: bool = False
    clarification_questions: list = field(default_factory=list)

    # Image
    image_disease: Optional[str] = None
    image_confidence: float = 0.0

    # Weather
    weather_summary: str = ""
    weather_risk: list = field(default_factory=list)

    # Final answer
    answer: str = ""
    primary_disease: str = ""
    confidence: float = 0.0
    treatments: list = field(default_factory=list)
    reasoning: str = ""

    # Metadata
    processing_steps: list = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


class AgriQAPipeline:
    """
    Main pipeline for AgriMultimodalQA.
    Singleton pattern for API efficiency.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logger.info("Initializing AgriQA Pipeline...")

        # Initialize modules (lazy loading)
        self._intent_classifier: Optional[IntentClassifier] = None
        self._ner_model: Optional[NERModel] = None
        self._ambiguity_resolver: Optional[AmbiguityResolver] = None
        self._kg: Optional[AgriculturalKG] = None
        self._image_classifier = None
        self._fusion: Optional[MultimodalFusion] = None
        self._generator: Optional[AgriAnswerGenerator] = None

        logger.info("Pipeline ready (lazy loading enabled)")

    @property
    def intent_classifier(self) -> IntentClassifier:
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifier(
                model_path=self.config.intent_model_path
            )
        return self._intent_classifier

    @property
    def ner_model(self) -> NERModel:
        if self._ner_model is None:
            self._ner_model = NERModel(
                model_path=self.config.ner_model_path
            )
        return self._ner_model

    @property
    def ambiguity_resolver(self) -> AmbiguityResolver:
        if self._ambiguity_resolver is None:
            self._ambiguity_resolver = AmbiguityResolver()
        return self._ambiguity_resolver

    @property
    def kg(self) -> AgriculturalKG:
        if self._kg is None:
            self._kg = AgriculturalKG(kg_path=self.config.kg_path)
        return self._kg

    @property
    def fusion(self) -> MultimodalFusion:
        if self._fusion is None:
            self._fusion = MultimodalFusion()
        return self._fusion

    @property
    def generator(self) -> AgriAnswerGenerator:
        if self._generator is None:
            self._generator = AgriAnswerGenerator(
                model_path=self.config.generator_model_path
            )
        return self._generator

    def process(
        self,
        query: str,
        image_path: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        session_context: Optional[Dict] = None
    ) -> QueryResult:
        """
        Process a user query through the full pipeline.

        Args:
            query: User's question in any supported language
            image_path: Optional path to leaf image
            latitude, longitude: GPS coordinates for weather
            session_context: Previous conversation context

        Returns:
            QueryResult with answer and all intermediate signals
        """
        result = QueryResult(original_query=query)
        steps = []

        try:
            # ── Step 1: Language Detection ─────────────────────────
            lang_info = detect_language(query)
            result.language = lang_info["language"]
            steps.append(f"1. Language: {result.language} ({lang_info['method']})")
            logger.info(f"Language: {result.language}")

            # ── Step 2: Script Normalization ───────────────────────
            norm = normalize_text(query, language=result.language)
            normalized_query = norm["normalized"]
            steps.append(f"2. Normalized: {normalized_query[:50]}")

            # ── Step 3: Translation ───────────────────────────────
            if result.language != "en":
                trans = translate_to_english(normalized_query, result.language)
                english_query = trans["translated"]
                steps.append(f"3. Translated: {english_query[:50]}")
            else:
                english_query = normalized_query
                steps.append("3. Translation: Not needed (English)")
            result.translated_query = english_query

            # ── Step 4: Intent Classification ──────────────────────
            intent_result = self.intent_classifier.predict(english_query)
            result.intent = intent_result["intent"]
            result.intent_confidence = intent_result["confidence"]
            steps.append(f"4. Intent: {result.intent} ({result.intent_confidence:.2f})")

            # ── Step 5: NER ────────────────────────────────────────
            ner_result = self.ner_model.predict(english_query)
            result.crops = ner_result.get("crops", [])
            result.diseases = ner_result.get("diseases", [])
            result.symptoms = ner_result.get("symptoms", [])
            result.regions = ner_result.get("regions", [])
            steps.append(
                f"5. NER: crops={result.crops}, diseases={result.diseases}, "
                f"symptoms={result.symptoms}"
            )

            # ── Step 6: Ambiguity Resolution ───────────────────────
            amb = self.ambiguity_resolver.analyze(
                english_query, result.intent, ner_result, result.intent_confidence
            )
            result.is_ambiguous = amb.is_ambiguous
            result.clarification_questions = amb.clarification_questions
            
            # If an image is provided, the visual data resolves vague symptom or crop ambiguities
            if image_path:
                result.is_ambiguous = False
                result.clarification_questions = []

            steps.append(f"6. Ambiguous: {result.is_ambiguous}")

            if result.is_ambiguous and amb.severity > 0.8:
                # Only abort on extreme ambiguity. Kept here just for logging structure.
                pass

            # ── Step 7: Knowledge Graph Query ──────────────────────
            crop = result.crops[0] if result.crops else None
            symptom = result.symptoms[0] if result.symptoms else None
            disease = result.diseases[0] if result.diseases else None
            kg_result = self.kg.query(crop=crop, symptom=symptom, disease=disease)
            steps.append(f"7. KG: {len(kg_result.get('diseases_found', []))} diseases found")

            # ── Step 8: Weather (optional) ─────────────────────────
            weather_context = {}
            weather_risk = []
            if self.config.enable_weather:
                region = result.regions[0] if result.regions else "india"
                try:
                    weather_context = get_weather_context(
                        region=region, lat=latitude, lon=longitude,
                        days_back=self.config.weather_days_back
                    )
                    if "parsed" in weather_context:
                        weather_risk = assess_disease_risk_from_weather(
                            weather_context["parsed"],
                            crop=crop or "rice"
                        ).get("risks", [])
                    result.weather_summary = weather_context.get("summary", "")
                    result.weather_risk = weather_risk
                    steps.append(f"8. Weather: {result.weather_summary[:60]}")
                except Exception as e:
                    logger.warning(f"Weather fetch failed: {e}")
                    steps.append("8. Weather: Failed (proceeding without)")

            # ── Step 9: Image Classifier (optional) ───────────────
            image_disease = None
            image_crop = None
            image_confidence = 0.0
            if image_path and self.config.enable_image:
                try:
                    if self._image_classifier is None:
                        from image.disease_classifier import DiseaseClassifier
                        self._image_classifier = DiseaseClassifier(
                            model_path=self.config.image_model_path
                        )
                    img_result = self._image_classifier.predict(image_path)
                    pred = img_result.get("top_prediction")
                    if pred:
                        image_disease = pred.get("disease")
                        image_crop = pred.get("crop")
                        image_confidence = pred.get("confidence", 0.0)
                        result.image_disease = image_disease
                        result.image_confidence = image_confidence
                    steps.append(f"9. Image: {image_disease} ({image_confidence:.0%})")
                except Exception as e:
                    logger.warning(f"Image classification failed: {e}")
                    steps.append("9. Image: Failed (proceeding without)")

            # ── Step 10: Multimodal Fusion ─────────────────────────
            fusion_input = FusionInput(
                query_text=english_query,
                intent=result.intent,
                intent_confidence=result.intent_confidence,
                crops=result.crops,
                diseases=result.diseases,
                symptoms=result.symptoms,
                regions=result.regions,
                language=result.language,
                image_disease=image_disease,
                image_crop=image_crop,
                image_confidence=image_confidence,
                weather_vector=weather_context.get("vector", [0.0] * 7),
                weather_summary=result.weather_summary,
                weather_risk=weather_risk,
                kg_diseases=kg_result.get("diseases_found", []),
                kg_treatments=kg_result.get("treatments", []),
                kg_triples=kg_result.get("triples", []),
                is_ambiguous=result.is_ambiguous,
                clarification_questions=result.clarification_questions,
            )
            fusion_output = self.fusion.fuse(fusion_input)
            result.primary_disease = fusion_output.primary_disease
            result.confidence = fusion_output.confidence
            result.treatments = fusion_output.treatments
            result.reasoning = fusion_output.reasoning
            steps.append(f"10. Fusion: disease={result.primary_disease} ({result.confidence:.0%})")

            # ── Step 11: Answer Generation ─────────────────────────
            gen_result = self.generator.generate(
                query=english_query,
                context=fusion_output.context
            )
            result.answer = gen_result["answer"]
            steps.append(f"11. Generated answer ({gen_result['method']})")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.error = str(e)
            result.answer = "An error occurred. Please try again."

        result.processing_steps = steps
        return result


# ── Singleton ─────────────────────────────────────────────────
_pipeline_instance: Optional[AgriQAPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> AgriQAPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AgriQAPipeline(config=config)
    return _pipeline_instance


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AgriMultimodalQA Pipeline Test")
    print("=" * 60)

    pipeline = AgriQAPipeline()

    test_queries = [
        "My rice plants in Tamil Nadu have yellow spots",
        "What fertilizer should I use for wheat?",
        "How to control aphids in cotton?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = pipeline.process(query)
        print(f"  Language : {result.language}")
        print(f"  Intent   : {result.intent} ({result.intent_confidence:.2f})")
        print(f"  Crops    : {result.crops}")
        print(f"  Diseases : {result.diseases}")
        print(f"  Symptoms : {result.symptoms}")
        print(f"  Disease  : {result.primary_disease}")
        print(f"  Answer   : {result.answer[:200]}")
        print(f"  Steps    : {len(result.processing_steps)} steps")
