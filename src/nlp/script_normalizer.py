"""
script_normalizer.py
--------------------
Normalizes Indian language scripts and mixed-language text.
Handles:
  - Unicode normalization (NFC/NFKC)
  - Whitespace cleanup
  - Punctuation normalization
  - Code-mixed text handling (Tanglish, Hinglish, etc.)
"""

import re
import unicodedata
from typing import Optional


# Common agricultural terms in transliterated form → keep as-is
AGRI_TRANSLITERATIONS = {
    "paddy": "paddy", "arisi": "rice", "gothumai": "wheat",
    "nellam": "paddy", "cholam": "sorghum", "tuvaram": "tur dal",
}


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Apply Unicode normalization.
    NFC = Canonical Decomposition followed by Canonical Composition
    NFKC = Compatibility decomposition + composition (strips ligatures)
    """
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Remove extra spaces, tabs, newlines."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_punctuation(text: str) -> str:
    """
    Normalize various Unicode punctuation to ASCII equivalents.
    Handles smart quotes, em-dashes, etc.
    """
    replacements = {
        '\u2018': "'", '\u2019': "'",  # Smart single quotes
        '\u201C': '"', '\u201D': '"',  # Smart double quotes
        '\u2013': '-', '\u2014': '-',  # En-dash, em-dash
        '\u2026': '...',               # Ellipsis
        '\u00A0': ' ',                 # Non-breaking space
        '\u200B': '',                  # Zero-width space
        '\u200C': '',                  # Zero-width non-joiner
        '\u200D': '',                  # Zero-width joiner (keep for some scripts?)
        '\uFEFF': '',                  # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_control_characters(text: str) -> str:
    """Remove non-printable control characters."""
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t ')


def normalize_digits(text: str) -> str:
    """
    Convert various Unicode digit forms to ASCII digits.
    Handles Tamil digits (௧, ௨...), Hindi digits (१, २...), etc.
    """
    # Tamil digits
    tamil_digits = '௦௧௨௩௪௫௬௭௮௯'
    # Devanagari digits
    deva_digits = '०१२३४५६७८९'
    # Telugu digits
    telugu_digits = '౦౧౨౩౪౫౬౭౮౯'

    for i, digit in enumerate(tamil_digits):
        text = text.replace(digit, str(i))
    for i, digit in enumerate(deva_digits):
        text = text.replace(digit, str(i))
    for i, digit in enumerate(telugu_digits):
        text = text.replace(digit, str(i))
    return text


def detect_code_mixing(text: str) -> dict:
    """
    Detect if text is code-mixed (e.g., Tanglish: Tamil + English in Latin script).
    Returns mixing ratio.
    """
    total_chars = len(text.replace(' ', ''))
    if total_chars == 0:
        return {"is_mixed": False, "latin_ratio": 0.0}

    latin_chars = sum(1 for c in text if '\u0000' <= c <= '\u007F' and c.isalpha())
    latin_ratio = latin_chars / total_chars

    # If mostly Latin but has Indian vocabulary → code-mixed
    is_mixed = 0.2 < latin_ratio < 0.8
    return {"is_mixed": is_mixed, "latin_ratio": round(latin_ratio, 2)}


def normalize_text(text: str, language: str = "en") -> dict:
    """
    Main normalization pipeline.

    Args:
        text (str): Raw input text
        language (str): Detected language code (ta/te/hi/en)

    Returns:
        dict: {
            "normalized": str,
            "original": str,
            "is_code_mixed": bool,
            "changes_made": list
        }
    """
    original = text
    changes = []

    # Step 1: Unicode normalization
    text_new = normalize_unicode(text, form="NFC")
    if text_new != text:
        changes.append("unicode_normalization")
    text = text_new

    # Step 2: Remove control characters
    text_new = remove_control_characters(text)
    if text_new != text:
        changes.append("control_char_removal")
    text = text_new

    # Step 3: Punctuation normalization
    text_new = normalize_punctuation(text)
    if text_new != text:
        changes.append("punctuation_normalization")
    text = text_new

    # Step 4: Digit normalization
    text_new = normalize_digits(text)
    if text_new != text:
        changes.append("digit_normalization")
    text = text_new

    # Step 5: Whitespace cleanup
    text_new = normalize_whitespace(text)
    if text_new != text:
        changes.append("whitespace_cleanup")
    text = text_new

    # Step 6: Detect code-mixing
    mixing_info = detect_code_mixing(text)

    return {
        "normalized": text,
        "original": original,
        "is_code_mixed": mixing_info["is_mixed"],
        "latin_ratio": mixing_info["latin_ratio"],
        "changes_made": changes,
        "language": language
    }


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("My rice plants have  yellow spots  ", "en"),
        ("நெல் செடிகளில்  மஞ்சள்  புள்ளிகள்", "ta"),
        ("Paddy la yellow spots irukku", "ta"),  # Tanglish
        ("धान के पौधों में \u200Bपीले धब्बे हैं", "hi"),
    ]
    print("=" * 60)
    print("Script Normalization Test")
    print("=" * 60)
    for text, lang in test_cases:
        result = normalize_text(text, lang)
        print(f"Original   : '{result['original']}'")
        print(f"Normalized : '{result['normalized']}'")
        print(f"Code-mixed : {result['is_code_mixed']} (Latin ratio: {result['latin_ratio']})")
        print(f"Changes    : {result['changes_made']}")
        print("-" * 40)
