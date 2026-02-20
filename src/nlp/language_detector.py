"""
language_detector.py
--------------------
Detects the language of an input text using Google Translate API (primary)
with Unicode script-range as an instant fallback.

Supports: Tamil (ta), Telugu (te), Hindi (hi), Kannada (kn),
          Malayalam (ml), Marathi (mr), English (en)
"""

from typing import Optional
from loguru import logger

# ── Google Translate auto-detect (primary) ────────────────────
try:
    from deep_translator import GoogleTranslator
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False
    logger.warning("deep-translator not installed. Using script-range detection.")

# ── Fallback: langdetect ──────────────────────────────────────
try:
    from langdetect import detect as _ld_detect, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# ── Unicode script ranges (instant, no network) ───────────────
SCRIPT_RANGES = {
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "te": (0x0C00, 0x0C7F),   # Telugu
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "ml": (0x0D00, 0x0D7F),   # Malayalam
    "hi": (0x0900, 0x097F),   # Devanagari (Hindi + Marathi share this block)
    "mr": (0x0900, 0x097F),   # Marathi (Devanagari — resolved by Google)
}

# Languages we explicitly support
SUPPORTED_LANGS = {"ta", "te", "hi", "kn", "ml", "mr", "en"}

# Google Translate language tag → our code
GOOGLE_TO_CODE = {
    "ta": "ta",
    "te": "te",
    "hi": "hi",
    "kn": "kn",
    "ml": "ml",
    "mr": "mr",
    "en": "en",
    # Some Google tags map to the same script
    "bn": "en",  # Bengali → not supported, treat as English
}


def _detect_by_script(text: str) -> Optional[str]:
    """
    Instant Unicode script detection.
    Reliable for non-Latin scripts. Returns None for Latin / ambiguous.
    Note: hi and mr share Devanagari — Google API resolves the difference.
    """
    counts: dict = {}
    for char in text:
        cp = ord(char)
        for lang, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1

    if not counts:
        return None  # Latin / ASCII → likely English

    # Tamil, Telugu, Kannada, Malayalam have unique blocks
    for lang in ("ta", "te", "kn", "ml"):
        if counts.get(lang, 0) > 0:
            return lang

    # Devanagari detected — could be hi or mr; return "hi" as placeholder
    # Google will give the exact language below
    if counts.get("hi", 0) > 0:
        return "deva"  # sentinel — Google will resolve

    return None


def _detect_by_google(text: str) -> Optional[str]:
    """Use Google Translate to auto-detect language."""
    if not _GOOGLE_AVAILABLE:
        return None
    try:
        # GoogleTranslator returns the detected source language when source='auto'
        t = GoogleTranslator(source="auto", target="en")
        translated = t.translate(text)
        # Access detected language through the internal attribute
        detected = getattr(t, "_source", None) or getattr(t, "source", None)
        if detected and detected != "auto":
            code = GOOGLE_TO_CODE.get(detected, detected)
            if code in SUPPORTED_LANGS:
                logger.info(f"Google detected language: {detected} → {code}")
                return code
    except Exception as e:
        logger.warning(f"Google language detection failed: {e}")
    return None


def _detect_by_langdetect(text: str) -> Optional[str]:
    """Fallback: langdetect library."""
    if not _LANGDETECT_AVAILABLE:
        return None
    try:
        lang = _ld_detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return None


def detect_language(text: str) -> dict:
    """
    Detect the language of input text.

    Pipeline:
      1. Unicode script range  → instant, no network
      2. Google Translate API  → for Devanagari (hi vs mr) + confirmation
      3. langdetect fallback
      4. Default English

    Returns:
        dict: { "language": str, "method": str, "confidence": str }
    """
    text = text.strip()
    if not text:
        return {"language": "en", "method": "default", "confidence": "low"}

    # Step 1: Script-range detection — handles Tamil/Telugu/Kannada/Malayalam instantly
    script_lang = _detect_by_script(text)

    if script_lang and script_lang != "deva":
        # Clear unique script — Tamil, Telugu, Kannada, or Malayalam
        return {"language": script_lang, "method": "unicode_script", "confidence": "high"}

    # Step 2: Google API — resolves hi vs mr, and confirms Latin-script languages
    google_lang = _detect_by_google(text)
    if google_lang:
        return {"language": google_lang, "method": "google_translate", "confidence": "high"}

    # Step 3: If script was Devanagari but Google failed, guess Hindi
    if script_lang == "deva":
        return {"language": "hi", "method": "unicode_script_fallback", "confidence": "medium"}

    # Step 4: langdetect
    ld = _detect_by_langdetect(text)
    if ld:
        return {"language": ld, "method": "langdetect", "confidence": "medium"}

    # Default
    return {"language": "en", "method": "default", "confidence": "low"}


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("My rice leaves have yellow spots", "en"),
        ("என் நெல் இலைகளில் மஞ்சள் புள்ளிகள்", "ta"),
        ("నా వరి ఆకులపై పసుపు మచ్చలు", "te"),
        ("मेरे धान की पत्तियों पर पीले धब्बे हैं", "hi"),
        ("माझ्या भाताच्या पानांवर पिवळे डाग आहेत", "mr"),
        ("ನನ್ನ ಭತ್ತದ ಎಲೆಗಳಲ್ಲಿ ಹಳದಿ ಚುಕ್ಕೆಗಳಿವೆ", "kn"),
        ("എന്റെ നെൽ ചെടികൾക്ക് മഞ്ഞ പൊട്ടുകളുണ്ട്", "ml"),
    ]
    print("=" * 60)
    print("Language Detection Test (Google API primary)")
    print("=" * 60)
    for text, expected in tests:
        result = detect_language(text)
        status = "✅" if result["language"] == expected else "⚠️"
        print(f"{status} [{expected}→{result['language']}] via {result['method']}")
        print(f"   Text: {text[:55]}")
        print()
