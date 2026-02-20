"""
nasa_weather.py
---------------
Fetches agricultural weather data from NASA POWER API.

API Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point

Parameters retrieved:
  - T2M      : Temperature at 2m (°C)
  - T2M_MAX  : Max temperature (°C)
  - T2M_MIN  : Min temperature (°C)
  - RH2M     : Relative Humidity at 2m (%)
  - PRECTOTCORR : Precipitation corrected (mm/day)
  - WS2M     : Wind speed at 2m (m/s)
  - ALLSKY_SFC_SW_DWN : Solar radiation (kJ/m²/day)

Usage (FREE, no API key needed):
  GET https://power.larc.nasa.gov/api/temporal/daily/point
    ?parameters=T2M,RH2M,PRECTOTCORR
    &community=AG
    &longitude=80.27
    &latitude=13.08
    &start=20240101
    &end=20240107
    &format=JSON
"""

import os
import json
import time
import datetime
from typing import Optional, Dict, List, Tuple
from loguru import logger

try:
    import requests
    import numpy as np
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.warning("requests/numpy not available")

# ── Constants ────────────────────────────────────────────────
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

WEATHER_PARAMETERS = [
    "T2M",           # Temperature at 2m (°C)
    "T2M_MAX",       # Max temperature
    "T2M_MIN",       # Min temperature
    "RH2M",          # Relative Humidity (%)
    "PRECTOTCORR",   # Precipitation (mm)
    "WS2M",          # Wind speed (m/s)
    "ALLSKY_SFC_SW_DWN",  # Solar radiation
]

# ── Mock / Fallback Weather Data (Feb values for Indian regions) ──
# Used when NASA POWER API is unreachable (no internet / timeout)
MOCK_WEATHER_BY_REGION = {
    "tamil nadu":      {"mean": 27.4, "max": 33.2, "min": 21.8, "humidity": 74.0, "precipitation": 4.2,  "wind_speed": 3.1, "solar": 18500},
    "andhra pradesh": {"mean": 28.1, "max": 34.5, "min": 21.2, "humidity": 68.0, "precipitation": 3.5,  "wind_speed": 2.8, "solar": 19200},
    "telangana":      {"mean": 27.8, "max": 34.0, "min": 21.5, "humidity": 65.0, "precipitation": 2.8,  "wind_speed": 2.5, "solar": 19000},
    "karnataka":      {"mean": 25.6, "max": 31.8, "min": 19.4, "humidity": 70.0, "precipitation": 5.1,  "wind_speed": 2.9, "solar": 17800},
    "kerala":         {"mean": 29.2, "max": 33.5, "min": 24.9, "humidity": 82.0, "precipitation": 18.4, "wind_speed": 3.5, "solar": 16200},
    "maharashtra":    {"mean": 26.3, "max": 33.1, "min": 19.5, "humidity": 60.0, "precipitation": 1.8,  "wind_speed": 3.2, "solar": 20100},
    "gujarat":        {"mean": 25.0, "max": 32.4, "min": 17.6, "humidity": 55.0, "precipitation": 0.5,  "wind_speed": 4.1, "solar": 20800},
    "punjab":         {"mean": 14.2, "max": 20.5, "min": 7.9,  "humidity": 72.0, "precipitation": 22.1, "wind_speed": 2.6, "solar": 13500},
    "haryana":        {"mean": 14.8, "max": 21.2, "min": 8.4,  "humidity": 69.0, "precipitation": 18.3, "wind_speed": 2.4, "solar": 13800},
    "uttar pradesh": {"mean": 17.5, "max": 24.8, "min": 10.2, "humidity": 66.0, "precipitation": 12.5, "wind_speed": 2.2, "solar": 14600},
    "bihar":          {"mean": 18.9, "max": 26.3, "min": 11.5, "humidity": 68.0, "precipitation": 10.8, "wind_speed": 2.1, "solar": 15200},
    "west bengal":   {"mean": 22.4, "max": 28.7, "min": 16.1, "humidity": 73.0, "precipitation": 8.6,  "wind_speed": 2.3, "solar": 15800},
    "rajasthan":      {"mean": 20.1, "max": 28.6, "min": 11.6, "humidity": 44.0, "precipitation": 0.8,  "wind_speed": 4.8, "solar": 21500},
    "madhya pradesh": {"mean": 22.8, "max": 30.5, "min": 15.1, "humidity": 58.0, "precipitation": 2.1,  "wind_speed": 2.7, "solar": 18200},
    "india":          {"mean": 24.0, "max": 30.0, "min": 17.0, "humidity": 65.0, "precipitation": 6.0,  "wind_speed": 3.0, "solar": 17000},
}


def get_mock_weather(region: str, days_back: int = 7) -> dict:
    """
    Return realistic mock weather data for a region.
    Used as fallback when NASA POWER API is unreachable.
    """
    key = region.lower().strip()
    # Find matching key
    data = None
    for k, v in MOCK_WEATHER_BY_REGION.items():
        if k in key or key in k:
            data = v
            break
    if data is None:
        data = MOCK_WEATHER_BY_REGION["india"]

    today = datetime.date.today()
    start = today - datetime.timedelta(days=days_back)
    return {
        "temperature": {
            "mean": data["mean"],
            "max":  data["max"],
            "min":  data["min"],
        },
        "humidity":        data["humidity"],
        "precipitation":   data["precipitation"],
        "wind_speed":      data["wind_speed"],
        "solar_radiation": data["solar"],
        "date_range": {
            "start": start.isoformat(),
            "end":   (today - datetime.timedelta(days=1)).isoformat(),
        },
        "daily": [],
        "is_mock": True,
    }

# Indian city coordinates for region-based lookup
REGION_COORDINATES = {
    "punjab": (30.9010, 75.8573),
    "haryana": (29.0588, 76.0856),
    "uttar pradesh": (26.8467, 80.9462),
    "bihar": (25.0961, 85.3131),
    "west bengal": (22.9868, 87.8550),
    "andhra pradesh": (15.9129, 79.7400),
    "telangana": (18.1124, 79.0193),
    "karnataka": (15.3173, 75.7139),
    "kerala": (10.8505, 76.2711),
    "tamil nadu": (11.1271, 78.6569),
    "maharashtra": (19.7515, 75.7139),
    "gujarat": (22.2587, 71.1924),
    "rajasthan": (27.0238, 74.2179),
    "madhya pradesh": (22.9734, 78.6569),
    # Default: India center
    "india": (20.5937, 78.9629),
}

# Default location (Chennai, Tamil Nadu)
DEFAULT_LAT = 13.0827
DEFAULT_LON = 80.2707


def get_coordinates_for_region(region: str) -> Tuple[float, float]:
    """Get lat/lon for a region name."""
    region_lower = region.lower().strip()
    for key, coords in REGION_COORDINATES.items():
        if key in region_lower or region_lower in key:
            return coords
    logger.warning(f"Region '{region}' not found, using default (India center)")
    return REGION_COORDINATES["india"]


def format_date(date: datetime.date) -> str:
    """Format date as YYYYMMDD for NASA API."""
    return date.strftime("%Y%m%d")


def fetch_weather_data(
    lat: float,
    lon: float,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    days_back: int = 7
) -> dict:
    """
    Fetch weather data from NASA POWER API.

    Args:
        lat, lon: Coordinates
        start_date, end_date: Date range (default: last 7 days with lag offset)
        days_back: Days to look back if no dates provided

    Returns:
        dict: Raw NASA API response or error dict
    """
    if not _REQUESTS_AVAILABLE:
        return {"error": "requests library not available"}

    # NASA POWER has a ~7-day data lag. Use safe window.
    NASA_LAG_DAYS = 7
    if end_date is None:
        end_date = datetime.date.today() - datetime.timedelta(days=NASA_LAG_DAYS)
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=days_back - 1)

    params = {
        "parameters": ",".join(WEATHER_PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": format_date(start_date),
        "end": format_date(end_date),
        "format": "JSON"
    }

    try:
        logger.info(f"Fetching NASA POWER data for ({lat:.2f}, {lon:.2f}) [{format_date(start_date)} to {format_date(end_date)}]")
        response = requests.get(NASA_POWER_BASE_URL, params=params, timeout=2)
        response.raise_for_status()
        data = response.json()
        logger.info("NASA POWER data fetched successfully")
        return data
    except requests.exceptions.Timeout:
        logger.warning("NASA POWER API timed out — will use mock data")
        return {"error": "timeout"}
    except requests.exceptions.ConnectionError:
        logger.warning("NASA POWER API unreachable (no internet?) — will use mock data")
        return {"error": "no_connection"}
    except requests.exceptions.RequestException as e:
        logger.warning(f"NASA POWER request failed: {e} — will use mock data")
        return {"error": f"request_error: {str(e)}"}
    except json.JSONDecodeError:
        logger.warning("NASA POWER returned invalid JSON — will use mock data")
        return {"error": "invalid_json"}


def parse_weather_data(raw_data: dict) -> dict:
    """
    Parse raw NASA POWER response into clean weather dict.

    Returns:
        dict: {
            "temperature": {"mean": float, "max": float, "min": float},
            "humidity": float,
            "precipitation": float,
            "wind_speed": float,
            "solar_radiation": float,
            "date_range": {"start": str, "end": str},
            "daily": list
        }
    """
    if "error" in raw_data:
        return {"error": raw_data["error"]}

    try:
        props = raw_data["properties"]["parameter"]
    except (KeyError, TypeError):
        return {"error": "Unexpected NASA POWER response format"}

    def avg(param_key: str) -> Optional[float]:
        """Average of a parameter across all dates."""
        values = list(props.get(param_key, {}).values())
        values = [v for v in values if v != -999]  # NASA uses -999 for missing
        return round(sum(values) / len(values), 2) if values else None

    def total(param_key: str) -> Optional[float]:
        """Sum of a parameter (for precipitation)."""
        values = list(props.get(param_key, {}).values())
        values = [v for v in values if v != -999]
        return round(sum(values), 2) if values else None

    # Construct daily records
    dates = list(props.get("T2M", {}).keys())
    daily = []
    for date in dates:
        daily.append({
            "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
            "temp_mean": props.get("T2M", {}).get(date),
            "temp_max": props.get("T2M_MAX", {}).get(date),
            "temp_min": props.get("T2M_MIN", {}).get(date),
            "humidity": props.get("RH2M", {}).get(date),
            "precipitation": props.get("PRECTOTCORR", {}).get(date),
            "wind_speed": props.get("WS2M", {}).get(date),
        })

    parsed = {
        "temperature": {
            "mean": avg("T2M"),
            "max": avg("T2M_MAX"),
            "min": avg("T2M_MIN"),
        },
        "humidity": avg("RH2M"),
        "precipitation": total("PRECTOTCORR"),
        "wind_speed": avg("WS2M"),
        "solar_radiation": avg("ALLSKY_SFC_SW_DWN"),
        "date_range": {
            "start": daily[0]["date"] if daily else None,
            "end": daily[-1]["date"] if daily else None,
        },
        "daily": daily
    }

    # If all T2M values were -999 (NASA missing data), fall back to mock
    if parsed["temperature"]["mean"] is None:
        logger.warning("NASA POWER returned all -999 (missing data) — will use mock data")
        return {"error": "all_missing"}

    return parsed


def weather_to_vector(weather: dict) -> list:
    """
    Convert weather data to a normalized feature vector for fusion.

    Returns:
        list: 7-dimensional weather feature vector (all values in [0, 1])
    """
    if "error" in weather:
        return [0.0] * 7

    # Normalization ranges (agricultural context)
    def normalize(value, min_val, max_val):
        if value is None:
            return 0.5  # Neutral missing value
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    temp = weather.get("temperature", {})
    vector = [
        normalize(temp.get("mean"), -10, 50),     # Temperature mean
        normalize(temp.get("max"), -10, 55),       # Temperature max
        normalize(temp.get("min"), -20, 45),       # Temperature min
        normalize(weather.get("humidity"), 0, 100), # Humidity
        normalize(weather.get("precipitation"), 0, 150),  # Precipitation
        normalize(weather.get("wind_speed"), 0, 20),      # Wind speed
        normalize(weather.get("solar_radiation"), 0, 30000),  # Solar radiation
    ]
    return [round(v, 4) for v in vector]


def get_weather_context(
    region: str = "india",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    days_back: int = 7
) -> dict:
    """
    High-level function to get weather for a region.

    Args:
        region: Region name (e.g., "Tamil Nadu")
        lat, lon: Explicit coordinates (overrides region)
        days_back: How many past days to average

    Returns:
        dict: Parsed weather + feature vector + summary text
    """
    if lat is None or lon is None:
        lat, lon = get_coordinates_for_region(region)

    raw = fetch_weather_data(lat, lon, days_back=days_back)
    parsed = parse_weather_data(raw)

    if "error" in parsed:
        # Fall back to mock data instead of returning an error
        logger.info(f"NASA API unavailable ({parsed['error']}), using mock weather for region='{region}'")
        parsed = get_mock_weather(region, days_back=days_back)

    vector = weather_to_vector(parsed)

    # Generate human-readable summary
    temp = parsed["temperature"]
    is_mock = parsed.get("is_mock", False)
    mock_note = " (estimated — NASA API offline)" if is_mock else ""
    summary = (
        f"Recent weather (last {days_back} days){mock_note}: "
        f"Avg temp {temp.get('mean')}°C (max: {temp.get('max')}°C, min: {temp.get('min')}°C), "
        f"Humidity: {parsed.get('humidity')}%, "
        f"Total rainfall: {parsed.get('precipitation')} mm."
    )

    return {
        "parsed": parsed,
        "vector": vector,
        "summary": summary,
        "region": region,
        "coordinates": {"latitude": lat, "longitude": lon},
        "is_mock": is_mock,
    }


def assess_disease_risk_from_weather(weather: dict, crop: str = "rice") -> dict:
    """
    Assess disease risk based on weather conditions.

    High humidity + moderate temp → Fungal disease risk
    Low humidity + high temp → Pest risk
    """
    if "error" in weather:
        return {"risk": "unknown", "reason": "Weather data unavailable"}

    humidity = weather.get("humidity", 50)
    temp_mean = weather.get("temperature", {}).get("mean", 25)
    precipitation = weather.get("precipitation", 0)

    risks = []

    if humidity > 80 and 20 <= temp_mean <= 35:
        risks.append({"disease": "Fungal blast/blight", "level": "HIGH", "reason": f"High humidity ({humidity}%) + optimal temp"})
    if precipitation > 50:
        risks.append({"disease": "Bacterial leaf blight", "level": "MEDIUM", "reason": f"High rainfall ({precipitation}mm)"})
    if temp_mean > 35 and humidity < 50:
        risks.append({"disease": "Pest infestation", "level": "HIGH", "reason": f"Hot & dry conditions"})

    if not risks:
        risks.append({"disease": "None significant", "level": "LOW", "reason": "Weather conditions are normal"})

    return {"crop": crop, "risks": risks, "weather_summary": weather}


# ── Quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("NASA POWER Weather Test")
    print("=" * 60)

    # Test with Tamil Nadu coordinates
    result = get_weather_context(region="Tamil Nadu", days_back=7)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Region     : {result['region']}")
        print(f"Coordinates: {result['coordinates']}")
        print(f"Summary    : {result['summary']}")
        print(f"Vector     : {result['vector']}")

        # Risk assessment
        risk = assess_disease_risk_from_weather(result["parsed"], crop="rice")
        print(f"\nDisease Risk Assessment:")
        for r in risk["risks"]:
            print(f"  [{r['level']}] {r['disease']} — {r['reason']}")
