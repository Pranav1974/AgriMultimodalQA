"""
main.py (FastAPI Backend)
-------------------------
REST API for AgriMultimodalQA System.

Endpoints:
  POST /ask           - Text query
  POST /ask-with-image - Text + image query
  GET  /health        - Health check
  GET  /kg/query      - Knowledge graph query
  GET  /weather       - Weather for region
"""

import os
import sys
import io
import base64
from typing import Optional
from loguru import logger

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import AgriQAPipeline, PipelineConfig, get_pipeline
from kg.agri_kg import get_kg
from weather.nasa_weather import get_weather_context

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="AgriMultimodalQA API",
    description="Multilingual Weather-Aware Agricultural QA System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ────────────────────────────────────
class TextQueryRequest(BaseModel):
    query: str
    region: Optional[str] = "india"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    language: Optional[str] = None  # Auto-detect if None


class QueryResponse(BaseModel):
    original_query: str
    language: str
    translated_query: str
    intent: str
    intent_confidence: float
    crops: list
    diseases: list
    symptoms: list
    primary_disease: str
    confidence: float
    treatments: list
    weather_summary: str
    answer: str
    reasoning: str
    is_ambiguous: bool
    clarification_questions: list
    error: Optional[str] = None


class KGQueryResponse(BaseModel):
    crop: Optional[str]
    symptom: Optional[str]
    disease: Optional[str]
    diseases_found: list
    treatments: list
    triples: list


class WeatherResponse(BaseModel):
    region: str
    coordinates: dict
    summary: str
    weather_vector: list
    temperature: dict
    humidity: Optional[float]
    precipitation: Optional[float]
    is_mock: bool = False
    error: Optional[str] = None


# ── Pipeline initialization ───────────────────────────────────
pipeline: Optional[AgriQAPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Starting AgriMultimodalQA API...")
    config = PipelineConfig(
        enable_weather=True,
        enable_image=True,
    )
    pipeline = AgriQAPipeline(config=config)
    logger.info("API ready!")


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AgriMultimodalQA",
        "version": "2.0.0"
    }


@app.post("/ask", response_model=QueryResponse, tags=["QA"])
async def ask_text(request: TextQueryRequest):
    """
    Process a text query in any supported language.

    Supports Tamil, Telugu, Hindi, English.
    Automatically detects language and translates to English for processing.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.process(
            query=request.query,
            latitude=request.latitude,
            longitude=request.longitude,
        )
        return QueryResponse(
            original_query=result.original_query,
            language=result.language,
            translated_query=result.translated_query,
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            crops=result.crops,
            diseases=result.diseases,
            symptoms=result.symptoms,
            primary_disease=result.primary_disease,
            confidence=result.confidence,
            treatments=result.treatments,
            weather_summary=result.weather_summary,
            answer=result.answer,
            reasoning=result.reasoning,
            is_ambiguous=result.is_ambiguous,
            clarification_questions=result.clarification_questions,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-with-image", response_model=QueryResponse, tags=["QA"])
async def ask_with_image(
    query: str = Form(...),
    region: str = Form("india"),
    image: Optional[UploadFile] = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    """
    Process a query with an optional leaf image.

    Accepts multipart form with text query + image file.
    Image is used to validate NLP-detected disease.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    image_path = None
    temp_file = None

    try:
        # Save image to temp file if provided
        if image:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(image.filename)[-1]
            )
            content = await image.read()
            temp_file.write(content)
            temp_file.close()
            image_path = temp_file.name

        result = pipeline.process(
            query=query,
            image_path=image_path,
            latitude=latitude,
            longitude=longitude,
        )

        return QueryResponse(
            original_query=result.original_query,
            language=result.language,
            translated_query=result.translated_query,
            intent=result.intent,
            intent_confidence=result.intent_confidence,
            crops=result.crops,
            diseases=result.diseases,
            symptoms=result.symptoms,
            primary_disease=result.primary_disease,
            confidence=result.confidence,
            treatments=result.treatments,
            weather_summary=result.weather_summary,
            answer=result.answer,
            reasoning=result.reasoning,
            is_ambiguous=result.is_ambiguous,
            clarification_questions=result.clarification_questions,
            error=result.error,
        )

    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if temp_file and image_path and os.path.exists(image_path):
            os.unlink(image_path)


@app.get("/kg/query", response_model=KGQueryResponse, tags=["Knowledge Graph"])
async def kg_query(
    crop: Optional[str] = Query(None),
    symptom: Optional[str] = Query(None),
    disease: Optional[str] = Query(None)
):
    """Query the agricultural knowledge graph."""
    try:
        kg = get_kg()
        result = kg.query(crop=crop, symptom=symptom, disease=disease)
        return KGQueryResponse(
            crop=crop,
            symptom=symptom,
            disease=disease,
            diseases_found=result.get("diseases_found", []),
            treatments=result.get("treatments", []),
            triples=result.get("triples", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather", response_model=WeatherResponse, tags=["Weather"])
async def get_weather(
    region: str = Query("india", description="Region name (e.g., 'Tamil Nadu')"),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    days: int = Query(7, description="Days of weather history")
):
    """Get weather data from NASA POWER API."""
    try:
        weather = get_weather_context(region=region, lat=lat, lon=lon, days_back=days)
        if "error" in weather:
            return WeatherResponse(
                region=region, coordinates={}, summary="",
                weather_vector=[], temperature={}, humidity=None, precipitation=None,
                error=weather["error"]
            )
        parsed = weather.get("parsed", {})
        temp = parsed.get("temperature", {})
        return WeatherResponse(
            region=region,
            coordinates=weather.get("coordinates", {}),
            summary=weather.get("summary", ""),
            weather_vector=weather.get("vector", []),
            temperature=temp,
            humidity=parsed.get("humidity"),
            precipitation=parsed.get("precipitation"),
            is_mock=weather.get("is_mock", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diseases/{crop}", tags=["Knowledge Graph"])
async def get_crop_diseases(crop: str):
    """Get all diseases for a specific crop."""
    try:
        kg = get_kg()
        diseases = kg.get_diseases_for_crop(crop)
        return {"crop": crop, "diseases": diseases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/treatments/{disease}", tags=["Knowledge Graph"])
async def get_disease_treatments(disease: str):
    """Get recommended treatments for a disease."""
    try:
        kg = get_kg()
        treatments = kg.get_treatments_for_disease(disease)
        symptoms = kg.get_symptoms_for_disease(disease)
        return {
            "disease": disease,
            "treatments": treatments,
            "symptoms": symptoms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
