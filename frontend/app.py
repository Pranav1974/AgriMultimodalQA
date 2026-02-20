"""
app.py — AgriMultimodalQA Streamlit Frontend
Multilingual UI: Tamil, Telugu, Kannada, Malayalam, Hindi, English
"""

import os
import sys
import requests
import streamlit as st

st.set_page_config(
    page_title="AgriMultimodalQA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── UI Translations ────────────────────────────────────────────
UI_TEXT = {
    "en": {
        "title": "AgriMultimodalQA",
        "subtitle": "Multilingual · Weather-Aware · Agricultural Question Answering",
        "ask": "Ask Your Question",
        "placeholder": "Type your agricultural question here...",
        "upload": "Upload leaf image (optional)",
        "btn": "Get Answer",
        "answer_title": "Answer",
        "weather_title": "Current Weather",
        "kg_title": "Knowledge Graph",
        "region_label": "Your Region",
        "lang_label": "Language",
        "crop_label": "Crop",
        "symptom_label": "Symptom",
        "kg_btn": "Query",
        "disease_label": "Likely Disease",
        "treatment_label": "Treatments",
        "processing": "Processing...",
        "fetching_weather": "Fetching weather...",
        "nav_home": "Home",
        "nav_about": "About",
        "about_title": "About the Project",
        "answer_lang_note": "Answer is in English",
    },
    "ta": {
        "title": "விவசாய கேள்வி-பதில்",
        "subtitle": "பல மொழி · வானிலை-விழிப்புடன் · விவசாய கேள்வி-பதில் அமைப்பு",
        "ask": "உங்கள் கேள்வி கேளுங்கள்",
        "placeholder": "உங்கள் விவசாய கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்...",
        "upload": "இலை படம் பதிவேற்றவும் (விருப்பமானது)",
        "btn": "பதில் பெறுங்கள்",
        "answer_title": "பதில்",
        "weather_title": "தற்போதைய வானிலை",
        "kg_title": "அறிவு வரைபடம்",
        "region_label": "உங்கள் பகுதி",
        "lang_label": "மொழி",
        "crop_label": "பயிர்",
        "symptom_label": "அறிகுறி",
        "kg_btn": "தேடு",
        "disease_label": "சாத்தியமான நோய்",
        "treatment_label": "சிகிச்சைகள்",
        "processing": "செயலாக்கப்படுகிறது...",
        "fetching_weather": "வானிலை பெறப்படுகிறது...",
        "nav_home": "முகப்பு",
        "nav_about": "பற்றி",
        "about_title": "திட்டத்தைப் பற்றி",
        "answer_lang_note": "பதில் தமிழில் உள்ளது",
    },
    "te": {
        "title": "వ్యవసాయ ప్రశ్నోత్తరాలు",
        "subtitle": "బహుభాషా · వాతావరణ-అవగాహన · వ్యవసాయ ప్రశ్నోత్తర వ్యవస్థ",
        "ask": "మీ ప్రశ్న అడగండి",
        "placeholder": "మీ వ్యవసాయ ప్రశ్నను ఇక్కడ టైప్ చేయండి...",
        "upload": "ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి (ఐచ్ఛికం)",
        "btn": "సమాధానం పొందండి",
        "answer_title": "సమాధానం",
        "weather_title": "ప్రస్తుత వాతావరణం",
        "kg_title": "జ్ఞాన గ్రాఫ్",
        "region_label": "మీ ప్రాంతం",
        "lang_label": "భాష",
        "crop_label": "పంట",
        "symptom_label": "లక్షణం",
        "kg_btn": "శోధించు",
        "disease_label": "సంభావ్య వ్యాధి",
        "treatment_label": "చికిత్సలు",
        "processing": "ప్రాసెస్ అవుతోంది...",
        "fetching_weather": "వాతావరణం పొందుతోంది...",
        "nav_home": "హోమ్",
        "nav_about": "గురించి",
        "about_title": "ప్రాజెక్ట్ గురించి",
        "answer_lang_note": "సమాధానం తెలుగులో ఉంది",
    },
    "kn": {
        "title": "ಕೃಷಿ ಪ್ರಶ್ನೋತ್ತರ",
        "subtitle": "ಬಹುಭಾಷಾ · ಹವಾಮಾನ-ಜಾಗರೂಕ · ಕೃಷಿ ಪ್ರಶ್ನೋತ್ತರ ವ್ಯವಸ್ಥೆ",
        "ask": "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಕೇಳಿ",
        "placeholder": "ನಿಮ್ಮ ಕೃಷಿ ಪ್ರಶ್ನೆಯನ್ನು ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ...",
        "upload": "ಎಲೆ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (ಐಚ್ಛಿಕ)",
        "btn": "ಉತ್ತರ ಪಡೆಯಿರಿ",
        "answer_title": "ಉತ್ತರ",
        "weather_title": "ಪ್ರಸ್ತುತ ಹವಾಮಾನ",
        "kg_title": "ಜ್ಞಾನ ಗ್ರಾಫ್",
        "region_label": "ನಿಮ್ಮ ಪ್ರದೇಶ",
        "lang_label": "ಭಾಷೆ",
        "crop_label": "ಬೆಳೆ",
        "symptom_label": "ರೋಗಲಕ್ಷಣ",
        "kg_btn": "ಹುಡುಕು",
        "disease_label": "ಸಂಭಾವ್ಯ ರೋಗ",
        "treatment_label": "ಚಿಕಿತ್ಸೆಗಳು",
        "processing": "ಸಂಸ್ಕರಿಸಲಾಗುತ್ತಿದೆ...",
        "fetching_weather": "ಹವಾಮಾನ ಪಡೆಯಲಾಗುತ್ತಿದೆ...",
        "nav_home": "ಮುಖಪುಟ",
        "nav_about": "ಬಗ್ಗೆ",
        "about_title": "ಯೋಜನೆಯ ಬಗ್ಗೆ",
        "answer_lang_note": "ಉತ್ತರ ಕನ್ನಡದಲ್ಲಿದೆ",
    },
    "ml": {
        "title": "കാർഷിക ചോദ്യോത്തരം",
        "subtitle": "ബഹുഭാഷ · കാലാവസ്ഥ-ബോധം · കാർഷിക ചോദ്യോത്തര സംവിധാനം",
        "ask": "നിങ്ങളുടെ ചോദ്യം ചോദിക്കൂ",
        "placeholder": "കാർഷിക ചോദ്യം ഇവിടെ ടൈപ്പ് ചെയ്യൂ...",
        "upload": "ഇല ചിത്രം അപ്‌ലോഡ് ചെയ്യൂ (ഓപ്ഷണൽ)",
        "btn": "ഉത്തരം നേടൂ",
        "answer_title": "ഉത്തരം",
        "weather_title": "നിലവിലെ കാലാവസ്ഥ",
        "kg_title": "നോളജ് ഗ്രാഫ്",
        "region_label": "നിങ്ങളുടെ പ്രദേശം",
        "lang_label": "ഭാഷ",
        "crop_label": "വിള",
        "symptom_label": "ലക്ഷണം",
        "kg_btn": "തിരയൂ",
        "disease_label": "സാധ്യതയുള്ള രോഗം",
        "treatment_label": "ചികിത്സകൾ",
        "processing": "പ്രോസസ്സ് ചെയ്യുന്നു...",
        "fetching_weather": "കാലാവസ്ഥ ലഭിക്കുന്നു...",
        "nav_home": "ഹോം",
        "nav_about": "കുറിച്ച്",
        "about_title": "പ്രോജക്ടിനെ കുറിച്ച്",
        "answer_lang_note": "ഉത്തരം മലയാളത്തിലാണ്",
    },
    "hi": {
        "title": "कृषि प्रश्नोत्तर",
        "subtitle": "बहुभाषी · मौसम-जागरूक · कृषि प्रश्नोत्तर प्रणाली",
        "ask": "अपना प्रश्न पूछें",
        "placeholder": "यहाँ अपना कृषि प्रश्न लिखें...",
        "upload": "पत्ती की छवि अपलोड करें (वैकल्पिक)",
        "btn": "उत्तर पाएं",
        "answer_title": "उत्तर",
        "weather_title": "वर्तमान मौसम",
        "kg_title": "ज्ञान ग्राफ",
        "region_label": "आपका क्षेत्र",
        "lang_label": "भाषा",
        "crop_label": "फसल",
        "symptom_label": "लक्षण",
        "kg_btn": "खोजें",
        "disease_label": "संभावित रोग",
        "treatment_label": "उपचार",
        "processing": "प्रसंस्करण हो रहा है...",
        "fetching_weather": "मौसम प्राप्त हो रहा है...",
        "nav_home": "होम",
        "nav_about": "के बारे में",
        "about_title": "प्रोजेक्ट के बारे में",
        "answer_lang_note": "उत्तर हिंदी में है",
    },
    "mr": {
        "title": "कृषी प्रश्नोत्तरे",
        "subtitle": "बहुभाषिक · हवामान-जागरूक · कृषी प्रश्नोत्तर प्रणाली",
        "ask": "आपला प्रश्न विचारा",
        "placeholder": "आपला शेतीविषयक प्रश्न येथे टाइप करा...",
        "upload": "पानाचा फोटो अपलोड करा (ऐच्छिक)",
        "btn": "उत्तर मिळवा",
        "answer_title": "उत्तर",
        "weather_title": "सध्याचे हवामान",
        "kg_title": "ज्ञान आलेख",
        "region_label": "आपला प्रदेश",
        "lang_label": "भाषा",
        "crop_label": "पीक",
        "symptom_label": "लक्षण",
        "kg_btn": "शोधा",
        "disease_label": "संभाव्य रोग",
        "treatment_label": "उपचार",
        "processing": "प्रक्रिया सुरू आहे...",
        "fetching_weather": "हवामान मिळवत आहे...",
        "nav_home": "मुख्यपृष्ठ",
        "nav_about": "बद्दल",
        "about_title": "प्रकल्पाबद्दल",
        "answer_lang_note": "उत्तर मराठीत आहे",
    },
}

LANG_OPTIONS = {
    "English":               "en",
    "தமிழ் (Tamil)":         "ta",
    "తెలుగు (Telugu)":       "te",
    "ಕನ್ನಡ (Kannada)":       "kn",
    "മലയാളം (Malayalam)":    "ml",
    "हिंदी (Hindi)":         "hi",
    "मराठी (Marathi)":       "mr",
}

GOOGLE_LANG_CODE = {"en":"en","ta":"ta","te":"te","kn":"kn","ml":"ml","hi":"hi","mr":"mr"}



# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  footer{visibility:hidden}
  #MainMenu{visibility:hidden}
  [data-testid="stDeployButton"]{display:none!important}
  [data-testid="stToolbar"]{display:none!important}
  header[data-testid="stHeader"]{background:transparent}

  /* ── Permanently lock sidebar open — hide collapse button ── */
  [data-testid="collapsedControl"]{ display:none !important; }
  button[kind="headerNoPadding"]{ display:none !important; }
  section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"]{
    display:none !important;
  }

  /* ── Force sidebar to stay visible ── */
  section[data-testid="stSidebar"]{
    display: block !important;
    visibility: visible !important;
    min-width: 280px !important;
    background: #f8fdf8 !important;
    border-right: 2px solid #d4edda !important;
  }
  /* Show the sidebar collapse arrow */
  [data-testid="collapsedControl"]{
    display: block !important;
    visibility: visible !important;
  }
  /* Sidebar inner padding */
  section[data-testid="stSidebar"] > div:first-child{
    padding: 1rem 1rem 2rem 1rem !important;
  }
  /* Language radio buttons styling */
  section[data-testid="stSidebar"] .stRadio > div{
    gap: 4px;
  }
  section[data-testid="stSidebar"] .stRadio label{
    background: white;
    border: 1px solid #c8e6c9;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.88rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  section[data-testid="stSidebar"] .stRadio label:hover{
    background: #e8f5e9;
    border-color: #52b788;
  }

  .hero{
    background:linear-gradient(135deg,#1b4332,#2d6a4f,#52b788);
    color:white;padding:1.2rem 2rem;border-radius:14px;
    text-align:center;margin-bottom:1rem;
    box-shadow:0 6px 24px rgba(45,106,79,.3)
  }
  .hero h1{font-size:1.9rem;font-weight:800;margin:0}
  .hero p{font-size:.95rem;opacity:.9;margin:.3rem 0 0}

  .card{background:white;border-radius:10px;padding:1rem 1.2rem;
        margin:.5rem 0;border-left:4px solid #52b788;
        box-shadow:0 2px 10px rgba(0,0,0,.07)}
  .answer-card{background:linear-gradient(135deg,#f0fdf4,#dcfce7);
               border-left:4px solid #16a34a;font-size:1rem;line-height:1.7}
  .disease-card{background:linear-gradient(135deg,#fff3e0,#fff8f1);
                border-left:4px solid #f4a261}
  .weather-card{background:linear-gradient(135deg,#e0f2fe,#f0f9ff);
                border-left:4px solid #0284c7}
  .mock-badge{background:#f59e0b;color:white;font-size:.7rem;
              padding:1px 7px;border-radius:10px;margin-left:5px}
  .about-section h3{color:#2d6a4f;margin-bottom:.3rem}
  .about-section li{margin:.2rem 0}
  .tag{display:inline-block;background:#52b788;color:white;
       padding:1px 9px;border-radius:12px;font-size:.82rem;margin:2px}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────
def call_ask(query, region):
    try:
        r = requests.post(f"{API_BASE_URL}/ask",
                          json={"query": query, "region": region}, timeout=60)
        r.raise_for_status(); return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure backend is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}

def call_ask_image(query, image_bytes, filename, region):
    try:
        r = requests.post(f"{API_BASE_URL}/ask-with-image",
                          data={"query": query, "region": region},
                          files={"image": (filename, image_bytes, "image/jpeg")}, timeout=120)
        r.raise_for_status(); return r.json()
    except Exception as e:
        return {"error": str(e)}

def call_weather(region):
    try:
        r = requests.get(f"{API_BASE_URL}/weather", params={"region": region}, timeout=25)
        r.raise_for_status(); return r.json()
    except Exception as e:
        return {"error": str(e)}

def call_kg(crop=None, symptom=None):
    try:
        params = {}
        if crop:    params["crop"]    = crop
        if symptom: params["symptom"] = symptom
        r = requests.get(f"{API_BASE_URL}/kg/query", params=params, timeout=15)
        r.raise_for_status(); return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def translate_text(text: str, target_lang: str) -> str:
    """Translate answer to target language using Google Translate."""
    if target_lang == "en" or not text:
        return text
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        return translated or text
    except Exception:
        return text  # return English if translation fails


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:

    # ── App Title in sidebar ──────────────────────────────────
    import base64
    import os

    # try custom logo first
    logo_path = None
    if os.path.exists("frontend/logo.png"): logo_path = "frontend/logo.png"
    elif os.path.exists("frontend/logo.jpg"): logo_path = "frontend/logo.jpg"
    
    if logo_path:
        with open(logo_path, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode("utf-8")
            mime = "image/png" if logo_path.endswith("png") else "image/jpeg"
            img_src = f"data:{mime};base64,{b64_logo}"
        
        st.markdown(f"""
        <div style="text-align:center;padding:.5rem 0 1rem">
            <img src="{img_src}" alt="Custom Logo" style="width:100%;max-width:260px;height:auto;border-radius:8px;"/>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Fallback SVG Logo (Intricate leaf-circuit)
        LOGO_SVG = """<svg width="60" height="60" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M 50 90 C -10 90 -10 20 50 25 C 110 20 110 90 50 90 Z" fill="url(#grad)"/>
        <!-- Arrow going up -->
        <path d="M 50 23 L 50 8 M 40 18 L 50 6 L 60 18" stroke="#1b4332" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
        <!-- Circuit Lines -->
        <path d="M 35 90 C 35 70 45 60 65 50" stroke="#f0fdf4" stroke-width="3" fill="none" stroke-linecap="round"/>
        <circle cx="65" cy="50" r="4" fill="#f0fdf4"/>
        <path d="M 45 90 C 45 78 52 70 70 65" stroke="#f0fdf4" stroke-width="3" fill="none" stroke-linecap="round"/>
        <circle cx="70" cy="65" r="4" fill="#f0fdf4"/>
        <path d="M 55 90 L 55 75 L 42 62" stroke="#f0fdf4" stroke-width="3" fill="none" stroke-linecap="round"/>
        <circle cx="42" cy="62" r="4" fill="#f0fdf4"/>
        <defs>
            <linearGradient id="grad" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stop-color="#74c69d" />
                <stop offset="100%" stop-color="#1b4332" />
            </linearGradient>
        </defs>
        </svg>"""
        b64_logo = base64.b64encode(LOGO_SVG.encode('utf-8')).decode('utf-8')
        img_src = f"data:image/svg+xml;base64,{b64_logo}"
        
        st.markdown(f"""
        <div style="text-align:center;padding:.5rem 0 .2rem">
            <img src="{img_src}" alt="Logo" style="width:60px;height:auto;margin-bottom:10px;"/><br>
            <strong style="font-size:1.05rem;color:#2d6a4f">AgriMultimodalQA</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Language Selector ─────────────────────────────────────
    st.markdown("### Select Language")
    st.markdown("<small style='color:#666'>UI & Answer language changes automatically</small>",
                unsafe_allow_html=True)

    lang_name = st.radio(
        "Language",
        options=list(LANG_OPTIONS.keys()),
        index=0,
        key="lang_select",
        label_visibility="collapsed"
    )
    lang_code = LANG_OPTIONS[lang_name]
    T = UI_TEXT.get(lang_code, UI_TEXT["en"])

    # Show selected language badge
    
    st.markdown(f"""
    <div style="background:#e8f5e9;border-radius:8px;padding:.4rem .8rem;
                margin:.3rem 0;font-size:.9rem;color:#2d6a4f">
        <strong>Active: {lang_name}</strong>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Region Selector ───────────────────────────────────────
    st.markdown(f"### {T['region_label']}")
    region = st.selectbox(
        "Region",
        options=["Tamil Nadu","Andhra Pradesh","Telangana","Karnataka",
                 "Kerala","Maharashtra","Gujarat","Punjab","Haryana",
                 "Uttar Pradesh","Bihar","West Bengal","Rajasthan",
                 "Madhya Pradesh","India"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ── Weather in sidebar ────────────────────────────────────
    st.markdown(f"### {T['weather_title']}")
    with st.spinner(T["fetching_weather"]):
        wd = call_weather(region)

    if not wd.get("error"):
        temp = wd.get("temperature", {})
        is_mock = wd.get("is_mock", False)
        mock_badge = '<span class="mock-badge">Estimated</span>' if is_mock else ""
        
        lbl_hum = translate_text("Humidity", lang_code)
        lbl_rain = translate_text("Rain", lang_code)
        
        st.markdown(f"""
        <div class="weather-card card" style="padding:.7rem 1rem">
            <strong>{temp.get('mean','N/A')}°C</strong>{mock_badge}
            &nbsp;<small>({temp.get('min','N/A')}–{temp.get('max','N/A')}°C)</small><br>
            {lbl_hum}: {wd.get('humidity','N/A')}%
            &nbsp;|&nbsp;
            {lbl_rain}: {wd.get('precipitation','N/A')} mm
        </div>""", unsafe_allow_html=True)
        if is_mock:
            st.caption(translate_text("NASA API offline — typical values shown", lang_code))

    else:
        st.warning(translate_text("Weather unavailable", lang_code))

    st.markdown("---")

    # ── Knowledge Graph in sidebar ────────────────────────────
    st.markdown(f"### {T['kg_title']}")
    kg_crop = st.selectbox(T["crop_label"],
                           ["","rice","wheat","cotton","tomato","potato","maize"],
                           key="kg_crop_sidebar")
    kg_sym = st.text_input(T["symptom_label"],
                           placeholder="e.g. yellow spots", key="kg_sym_sidebar")
    if st.button(T["kg_btn"], key="kg_btn_sidebar"):
        kr = call_kg(crop=kg_crop or None, symptom=kg_sym or None)
        if not kr.get("error"):
            diseases = kr.get("diseases_found", [])
            if diseases:
                for d in diseases[:4]:
                    dtname = translate_text(d.get('name', d.get('id','')), lang_code)
                    st.markdown(f"• {dtname}")
            else:
                st.caption(translate_text("No diseases found", lang_code))
        else:
            st.caption(translate_text("KG unavailable", lang_code))

    st.markdown("---")




# ── Navigation ────────────────────────────────────────────────
page = st.radio("", [T["nav_home"], T["nav_about"]],
                horizontal=True, label_visibility="collapsed", key="nav")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════
if page == T["nav_home"]:

    # Hero banner
    st.markdown(f"""
    <div class="hero">
        <h1>{T['title']}</h1>
        <p>{T['subtitle']}</p>
    </div>""", unsafe_allow_html=True)

    # Query form
    st.markdown(f"### {T['ask']}")

    query = st.text_area(
        "", height=90,
        placeholder=T["placeholder"],
        key="main_query",
        label_visibility="collapsed"
    )

    uploaded = st.file_uploader(
        T["upload"], type=["jpg","jpeg","png","webp"], key="leaf_img"
    )
    if uploaded:
        st.image(uploaded, width=260)

    submit = st.button(T["btn"], type="primary", use_container_width=True, key="submit_btn")

    # ── Process Query ─────────────────────────────────────────
    if submit and (query.strip() or uploaded):
        if uploaded and not query.strip():
            query = "What is wrong with this plant?"
        
        with st.spinner(T["processing"]):
            if uploaded:
                uploaded.seek(0)
                image_bytes = uploaded.read()
                result = call_ask_image(query, image_bytes, uploaded.name, region)
            else:
                result = call_ask(query, region)

        if result.get("error"):
            st.error(f"{result['error']}")
        else:
            st.markdown(f"### {T['answer_title']}")

            # Translate answer into the SAME language the user typed in
            english_answer = result.get("answer", "")
            query_lang = result.get("language", "en")   # detected from the typed query

            LANG_NAMES = {
                "en": "English", "ta": "Tamil (தமிழ்)", "te": "Telugu (తెలుగు)",
                "kn": "Kannada (ಕನ್ನಡ)", "ml": "Malayalam (മലയാളം)",
                "hi": "Hindi (हिंदी)", "mr": "Marathi (मराठी)"
            }

            final_answer = translate_text(english_answer, query_lang)

            st.markdown(f"""
            <div class="answer-card card">
                {final_answer}
            </div>""", unsafe_allow_html=True)

            if query_lang != "en":
                lang_name_display = LANG_NAMES.get(query_lang, query_lang.upper())
                st.caption(f"Answer translated to **{lang_name_display}** (detected from your query)")

            # Compact disease + treatment card (if detected)
            disease = result.get("primary_disease", "")
            treatments = result.get("treatments", [])
            if disease and disease not in ("unknown", ""):
                conf = result.get("confidence", 0)
                emoji = "" if conf > 0.7 else "" if conf > 0.4 else ""
                treat_text = ""
                if treatments:
                    names = [t.get("name","") for t in treatments[:3] if t.get("name")]
                    treat_text = f"<br><strong>{T['treatment_label']}:</strong> {', '.join(names)}"
                st.markdown(f"""
                <div class="disease-card card">
                    <strong>{T['disease_label']}:</strong> {emoji} {disease.title()}
                    &nbsp;<small>({int(conf*100)}%)</small>
                    {treat_text}
                </div>""", unsafe_allow_html=True)


    elif submit:
        st.warning("" + T["placeholder"])


# ══════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════
else:
    st.markdown(f"## {T['about_title']}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="about-section card">
            <h3>{translate_text('Objective', lang_code)}</h3>
            <p>{translate_text('AgriMultimodalQA is a multimodal AI system that helps Indian farmers get accurate, language-aware answers to agricultural queries — combining text, images, weather data, and knowledge graphs into one unified system.', lang_code)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-section card">
            <h3>{translate_text('Tech Stack', lang_code)}</h3>
            <ul>
                <li><b>XLM-RoBERTa</b> — {translate_text('Multilingual NLP (Intent + NER)', lang_code)}</li>
                <li><b>EfficientNet-B3</b> — {translate_text('Leaf disease image classifier', lang_code)}</li>
                <li><b>NASA POWER API</b> — {translate_text('Live satellite weather data', lang_code)}</li>
                <li><b>NetworkX KG</b> — {translate_text('Agricultural Knowledge Graph', lang_code)}</li>
                <li><b>Flan-T5 (RAG)</b> — {translate_text('Answer generation', lang_code)}</li>
                <li><b>Multimodal Fusion</b> — {translate_text('Signal integration', lang_code)}</li>
                <li><b>FastAPI + Streamlit</b> — {translate_text('Backend & Frontend', lang_code)}</li>
                <li><b>Google Translate</b> — {translate_text('Instant multilingual support', lang_code)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="about-section card">
            <h3>{translate_text('Supported Languages', lang_code)}</h3>
            <span class="tag">English</span>
            <span class="tag">🇮🇳 Tamil (தமிழ்)</span>
            <span class="tag">🇮🇳 Telugu (తెలుగు)</span>
            <span class="tag">🇮🇳 Kannada (ಕನ್ನಡ)</span>
            <span class="tag">🇮🇳 Malayalam (മലയാളം)</span>
            <span class="tag">🇮🇳 Hindi (हिंदी)</span>
            <span class="tag">🇮🇳 Marathi (मराठी)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-section card">
            <h3>{translate_text('Key Features', lang_code)}</h3>
            <ul>
                <li>{translate_text('Automatic language detection & translation', lang_code)}</li>
                <li>{translate_text('Crop & disease Named Entity Recognition', lang_code)}</li>
                <li>{translate_text('Leaf image disease validation (EfficientNet)', lang_code)}</li>
                <li>{translate_text('Real-time NASA satellite weather integration', lang_code)}</li>
                <li>{translate_text('Knowledge graph-backed treatment advice', lang_code)}</li>
                <li>{translate_text('Answers in your chosen language', lang_code)}</li>
                <li>{translate_text('All South Indian languages supported', lang_code)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-section card">
            <h3>{translate_text('System Architecture', lang_code)}</h3>
            <p>
            {translate_text('Query → Language Detect → Translate → Intent + NER → KG Lookup → Weather Fetch → Image Analysis → Multimodal Fusion → RAG Answer → Translate Back', lang_code)}
            </p>
            <h3>{translate_text('Team', lang_code)}</h3>
            <p><strong>Pranav</strong> & <strong>Adithya</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#666;font-size:.82rem">
        AgriMultimodalQA &nbsp;|&nbsp;
        XLM-RoBERTa · EfficientNet-B3 · Flan-T5 · NASA POWER · NetworkX
    </div>
    """, unsafe_allow_html=True)
