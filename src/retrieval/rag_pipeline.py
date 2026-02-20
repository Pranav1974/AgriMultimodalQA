"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation pipeline for agricultural QA.

Pipeline:
  1. Take fused context (from MultimodalFusion)
  2. Retrieve relevant agriculture documents (FAISS vector search)
  3. Construct prompt with context + retrieved docs
  4. Generate answer using Flan-T5-small

Model: google/flan-t5-small (lightweight, ~80M params, runs on CPU)
Alternative: google/flan-t5-base for better quality
"""

import os
from typing import Optional, List, Dict
from loguru import logger

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Generator disabled.")

try:
    import numpy as np
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using BM25 retrieval.")

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# ── Agricultural Document Corpus (Built-in) ──────────────────
AGRI_CORPUS = [
    # Rice diseases
    "Rice blast (Magnaporthe oryzae) causes diamond-shaped lesions on leaves. Treat with Tricyclazole 75% WP at 0.6g per liter or Carbendazim at 1g per liter. Apply in evening. High humidity triggers this disease.",
    "Bacterial Leaf Blight in rice is caused by Xanthomonas oryzae. It shows water-soaked lesions that turn yellow-brown. Treat with Streptomycin + Tetracycline spray. Avoid waterlogging.",
    "Brown Spot in rice (Cochliobolus miyabeanus) causes brown circular spots with yellow halo. Spray Mancozeb at 2g per liter. Apply potassium fertilizer to reduce susceptibility.",
    "Sheath Blight (Rhizoctonia solani) in rice shows lesions at water level on sheath. Apply Carbendazim or Validamycin. Reduce nitrogen application and plant density.",
    "Rice Tungro Virus causes yellow-orange discoloration and stunted growth. No chemical cure. Control green leafhoppers that spread the virus using Chlorpyrifos.",

    # Wheat diseases
    "Wheat Yellow Rust (Puccinia striiformis) shows yellow-orange pustules in stripes on leaves. Spray Propiconazole or Mancozeb. Grow resistant varieties. Thrives in cool, moist conditions.",
    "Wheat Crown Rot causes browning and rotting at the base. Seed treatment with Thiram or Carboxin helps. Ensure well-drained soil.",

    # Tomato diseases
    "Tomato Late Blight (Phytophthora infestans) shows dark brown patches on leaves and stem with white mold. Spray Mancozeb or Copper Oxychloride at 2g per liter every 7-10 days.",
    "Tomato Early Blight (Alternaria solani) shows concentric ring spots on lower leaves. Use good crop rotation. Spray Mancozeb regularly. Remove infected leaves.",
    "Tomato Leaf Curl Virus is spread by whiteflies. Control whiteflies with Imidacloprid. Use disease-resistant tomato varieties. Remove and destroy infected plants.",

    # General fertilizer advice
    "Rice nitrogen management: Apply 120-150 kg N/ha in 3 splits - at basal, tillering, and panicle initiation stages. Use urea (46% N) or ammonium sulfate.",
    "Paddy requires NPK ratio of 100:50:50 kg/ha for high-yield varieties. Apply DAP for phosphorus, MOP for potassium.",
    "Wheat fertilizer: Apply 120-150 kg nitrogen, 60 kg phosphorus, 40 kg potassium per hectare. Split nitrogen into 2-3 doses.",
    "Cotton needs 100-120 kg N/ha. Apply 25% N at sowing, 25% at square formation, 50% at boll development.",

    # Pest management
    "Rice Brown Planthopper (Nilaparvata lugens) sucks sap from base of plants causing 'hopper burn'. Apply Buprofezin or Thiamethoxam. Drain fields periodically.",
    "Rice Stem Borer causes 'dead heart' (in vegetative stage) and 'white ear' (in reproductive stage). Apply Carbofuran granules at tillering. Use light traps.",
    "Cotton Bollworm: Spray Chlorpyrifos + Cypermethrin at 2.5ml per liter. Use pheromone traps for monitoring. Rotate insecticides to prevent resistance.",

    # Weather-based advice
    "When humidity is above 85% and temperature is between 20-30°C, fungal diseases like blast and blight are most active. Apply preventive fungicides.",
    "During drought stress, plants become more susceptible to spider mites and thrips. Irrigation and Neem oil spray (3ml/L) helps control them.",
    "Excessive rainfall (>100mm per week) can cause waterlogging and root rot. Drain fields immediately. Apply Captan fungicide preventively.",

    # General crop health
    "Integrated Pest Management (IPM): Use cultural control (crop rotation, resistant varieties), biological control (Trichoderma, Pseudomonas), and chemical control as last resort.",
    "Soil pH between 6.0-7.0 is ideal for most crops. Add lime to increase pH (for acidic soil) or sulfur to decrease pH (for alkaline soil).",
    "Neem-based products (Azadirachtin) are effective organic pesticides against many insects. Safe for beneficial insects like bees and earthworms.",
    "Biofertilizers: Rhizobium for legumes (fixes nitrogen), Azospirillum for cereals, PSB (Phosphate Solubilizing Bacteria) for all crops.",
]


class SimpleVectorStore:
    """
    Simple FAISS-based or BM25-based document retrieval.
    """

    def __init__(self, corpus: List[str], use_dense: bool = True):
        self.corpus = corpus
        self.use_dense = use_dense and _FAISS_AVAILABLE and _SBERT_AVAILABLE
        self._index = None
        self._encoder = None
        self._bm25 = None

        if self.use_dense:
            self._build_faiss_index()
        elif _BM25_AVAILABLE:
            self._build_bm25_index()
        else:
            logger.warning("No retrieval backend available. Using simple keyword search.")

    def _build_faiss_index(self):
        """Build FAISS index using sentence-transformers."""
        try:
            logger.info("Building FAISS vector index...")
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, small
            embeddings = self._encoder.encode(self.corpus, show_progress_bar=False)
            embeddings = embeddings.astype("float32")

            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalize)
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)
            logger.info(f"FAISS index built: {len(self.corpus)} documents")
        except Exception as e:
            logger.error(f"FAISS index failed: {e}")
            self.use_dense = False
            self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index."""
        if not _BM25_AVAILABLE:
            return
        tokenized = [doc.lower().split() for doc in self.corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k relevant documents."""
        if self.use_dense and self._index and self._encoder:
            return self._dense_retrieve(query, top_k)
        elif self._bm25:
            return self._bm25_retrieve(query, top_k)
        else:
            return self._keyword_retrieve(query, top_k)

    def _dense_retrieve(self, query: str, top_k: int) -> List[str]:
        query_emb = self._encoder.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        _, indices = self._index.search(query_emb, top_k)
        return [self.corpus[i] for i in indices[0] if 0 <= i < len(self.corpus)]

    def _bm25_retrieve(self, query: str, top_k: int) -> List[str]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.corpus[i] for i in top_indices]

    def _keyword_retrieve(self, query: str, top_k: int) -> List[str]:
        """Simple keyword overlap fallback."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:top_k] if _ > 0]


class AgriAnswerGenerator:
    """
    Flan-T5-small based answer generator for agricultural QA.
    """

    MODEL_NAME = "google/flan-t5-small"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        corpus: Optional[List[str]] = None
    ):
        corpus = corpus or AGRI_CORPUS
        self.retriever = SimpleVectorStore(corpus)

        if not _TORCH_AVAILABLE:
            self.model = None
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        load_path = model_path or self.MODEL_NAME

        try:
            logger.info(f"Loading Flan-T5 from {load_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = T5ForConditionalGeneration.from_pretrained(load_path)
            self.model.eval()
            self.model.to(self.device)
            logger.info("Flan-T5 loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Flan-T5: {e}")
            self.model = None

    def build_prompt(
        self, context: str, retrieved_docs: List[str], query: str
    ) -> str:
        """Construct a structured prompt for the T5 model."""
        docs_text = "\n".join([f"- {doc}" for doc in retrieved_docs[:3]])
        prompt = (
            f"You are an expert agricultural assistant. Answer the farmer's question accurately.\n\n"
            f"Context information:\n{context}\n\n"
            f"Reference knowledge:\n{docs_text}\n\n"
            f"Farmer's question: {query}\n\n"
            f"Provide a clear, practical answer:"
        )
        return prompt

    def generate(
        self,
        query: str,
        context: str,
        max_length: int = 300,
        num_beams: int = 4
    ) -> dict:
        """
        Generate an answer for the query.

        Args:
            query: The user's question (in English)
            context: Fused context from MultimodalFusion
            max_length: Max tokens in generated answer
            num_beams: Beam search width

        Returns:
            dict: {
                "answer": str,
                "retrieved_docs": list,
                "prompt_used": str
            }
        """
        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(query, top_k=3)

        if self.model is None:
            # Template-based fallback
            answer = self._template_answer(query, context, retrieved)
            return {
                "answer": answer,
                "retrieved_docs": retrieved,
                "prompt_used": "template_fallback",
                "method": "template"
            }

        # Build prompt
        prompt = self.build_prompt(context, retrieved, query)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "answer": answer,
                "retrieved_docs": retrieved,
                "prompt_used": prompt[:200] + "...",
                "method": "flan_t5"
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "answer": self._template_answer(query, context, retrieved),
                "retrieved_docs": retrieved,
                "prompt_used": "error_fallback",
                "method": "template"
            }

    def _template_answer(
        self, query: str, context: str, retrieved: List[str]
    ) -> str:
        """Rule-based template answer when model unavailable."""
        if "Clarification needed:" in context:
            clarification = context.split("Clarification needed: ")[1]
            return f"**Clarification needed:** {clarification}"

        lines = context.split("\n")
        
        disease_raw = next((l for l in lines if "Likely disease: " in l), None)
        disease = disease_raw.split("Likely disease: ")[1].split(" (confidence")[0] if disease_raw else None
        
        crop_raw = next((l for l in lines if "Crop: " in l), None)
        crop = crop_raw.split("Crop: ")[1] if crop_raw else "your crop"
        
        treatment_raw = next((l for l in lines if "Recommended treatments: " in l), None)
        treatment = treatment_raw.split("Recommended treatments: ")[1] if treatment_raw else None

        parts = []

        if disease and disease.lower() not in ("unknown", "none", ""):
            parts.append(f"Based on our analysis, **{crop}** may be affected by **{disease}**.")
            if treatment:
                parts.append(f"**Recommended treatment:** {treatment}.")
        else:
            parts.append("Here is some information related to your query:")

        # Always append the top retrieved knowledge docs
        if retrieved:
            parts.append("\n**Reference information:**")
            for doc in retrieved[:2]:
                parts.append(f"• {doc}")

        if len(parts) <= 1 and not retrieved:
            parts = [
                "Please provide more details about the crop type and specifically what you need help with "
                "(e.g. fertilizer, symptoms like spots or wilting) for a better answer."
            ]

        return "\n\n".join(parts)


# Singleton
_generator_instance: Optional[AgriAnswerGenerator] = None


def get_generator(model_path: Optional[str] = None) -> AgriAnswerGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = AgriAnswerGenerator(model_path=model_path)
    return _generator_instance


def generate_answer(query: str, context: str) -> dict:
    return get_generator().generate(query, context)


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Answer Generation Test")
    print("=" * 60)

    generator = AgriAnswerGenerator(model_path=None)  # Uses template fallback

    test_query = "My rice plants have yellow spots and I'm in Tamil Nadu with high humidity"
    test_context = """User query: My rice plants have yellow spots and I'm in Tamil Nadu with high humidity
Intent: DISEASE_DIAGNOSIS
Crop: rice
Observed symptoms: yellow spots
Likely disease: rice blast (confidence: 85%)
Image analysis: Rice with Rice Blast (78% confidence)
Weather: High humidity (87%), temp 28°C, rainfall 65mm
Knowledge graph: Rice Blast AFFECTS Rice | Rice Blast TREATED_BY Tricyclazole
Recommended treatments: Tricyclazole 75% WP, Carbendazim 50% WP"""

    result = generator.generate(test_query, test_context)
    print(f"Query  : {test_query}")
    print(f"Method : {result['method']}")
    print(f"Answer : {result['answer']}")
    print(f"\nTop retrieved doc: {result['retrieved_docs'][0][:150]}...")
