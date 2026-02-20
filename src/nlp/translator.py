"""
translator.py
-------------
Translates Tamil / Telugu / Hindi queries to English.

Primary  : deep-translator (GoogleTranslator) — free, no model download, instant
Fallback : keyword-based approximation if internet unavailable
"""

import os
from typing import Optional, Dict
from loguru import logger

try:
    from deep_translator import GoogleTranslator
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False
    logger.warning("deep-translator not installed. Run: pip install deep-translator")

# Language code → Google Translate language code
LANG_CODE_MAP = {
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "hi": "hi",   # Hindi
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "mr": "mr",   # Marathi
    "en": "en",
}

# ── Keyword-based fallback (when offline) ─────────────────────
# Maps common agricultural words in each language → English
KEYWORD_MAP = {
    # Tamil
    "நெல்": "rice", "வரி": "rice", "நெல் செடி": "rice plant",
    "கோதுமை": "wheat", "பருத்தி": "cotton", "தக்காளி": "tomato",
    "மஞ்சள்": "yellow", "புள்ளிகள்": "spots", "நோய்": "disease",
    "இலை": "leaf", "இலைகள்": "leaves", "வேர்": "root",
    "பூஞ்சை": "fungal", "பூச்சி": "pest", "ஒட்டுண்ணி": "parasite",
    "சிகிச்சை": "treatment", "மருந்து": "medicine", "உரம்": "fertilizer",
    "வெண்மை": "white", "கருப்பு": "black", "பழுப்பு": "brown",
    "வாடுகிறது": "wilting", "காய்ந்து": "drying", "விழுகிறது": "falling",
    "என்": "my", "என்ன": "what", "எப்படி": "how",
    # Telugu
    "వరి": "rice", "గోధుమ": "wheat", "పత్తి": "cotton",
    "టొమాటో": "tomato", "మొక్కజొన్న": "maize",
    "పసుపు": "yellow", "మచ్చలు": "spots", "వ్యాధి": "disease",
    "ఆకు": "leaf", "ఆకులు": "leaves", "వేర్లు": "roots",
    "శిలీంధ్రం": "fungal", "పురుగు": "pest",
    "చికిత్స": "treatment", "ఎరువు": "fertilizer",
    "తెల్లగా": "white", "నల్లగా": "black", "గోధుమ": "brown",
    "వాడిపోతున్న": "wilting", "ఎండిపోతున్న": "drying",
    "నా": "my", "ఏమిటి": "what", "ఎలా": "how",
    # Hindi
    "धान": "rice", "चावल": "rice", "गेहूं": "wheat",
    "कपास": "cotton", "टमाटर": "tomato", "मक्का": "maize",
    "पीले": "yellow", "धब्बे": "spots", "रोग": "disease",
    "पत्ती": "leaf", "पत्तियां": "leaves",
    "फफूंद": "fungal", "कीट": "pest",
    "उपचार": "treatment", "खाद": "fertilizer",
    "सफेद": "white", "काला": "black", "भूरा": "brown",
    "मुरझाना": "wilting",
    "मेरे": "my", "क्या": "what", "कैसे": "how",
}


def _keyword_fallback(text: str) -> str:
    """Very rough keyword substitution when translation APIs unavailable."""
    result = text
    for native, english in KEYWORD_MAP.items():
        result = result.replace(native, english)
    # If unchanged, just return as-is (pipeline will still try to match)
    return result


def translate_to_english(text: str, src_lang: str) -> dict:
    """
    Translate text from src_lang to English.

    Args:
        text: Input text
        src_lang: Language code (ta / te / hi / en)

    Returns:
        dict with keys: translated, original, src_lang, tgt_lang, success
    """
    if src_lang == "en" or not text.strip():
        return {
            "translated": text,
            "original": text,
            "src_lang": "en",
            "tgt_lang": "en",
            "success": True,
        }

    # ── Primary: Google Translate (deep-translator) ────────────
    if _GOOGLE_AVAILABLE:
        try:
            google_lang = LANG_CODE_MAP.get(src_lang, src_lang)
            translator = GoogleTranslator(source=google_lang, target="en")
            translated = translator.translate(text)
            if translated:
                logger.info(f"Google Translate [{src_lang}→en]: {translated[:80]}")
                return {
                    "translated": translated,
                    "original": text,
                    "src_lang": src_lang,
                    "tgt_lang": "en",
                    "success": True,
                }
        except Exception as e:
            logger.warning(f"Google Translate failed ({e}), using keyword fallback")

    # ── Fallback: keyword substitution ────────────────────────
    fallback = _keyword_fallback(text)
    logger.warning(f"Using keyword fallback translation: {fallback[:80]}")
    return {
        "translated": fallback,
        "original": text,
        "src_lang": src_lang,
        "tgt_lang": "en",
        "success": False,
    }


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("என் நெல் இலைகள் மஞ்சள் நிறமாக மாறுகிறது. இது என்ன நோய்?", "ta"),
        ("నా వరి మొక్కలకు పసుపు మచ్చలు వచ్చాయి", "te"),
        ("धान की पत्तियों पर पीले धब्बे हैं", "hi"),
        ("My rice plants have yellow spots", "en"),
    ]
    print("=" * 60)
    print("Translation Test (Google Translate via deep-translator)")
    print("=" * 60)
    for text, lang in tests:
        result = translate_to_english(text, lang)
        print(f"[{lang}] {result['original'][:50]}")
        print(f"  → {result['translated']}")
        print(f"  success={result['success']}")
        print()
