@echo off
echo ============================================================
echo  AgriMultimodalQA - Start Script
echo ============================================================
echo.

REM Check Python
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Install Python 3.9+
    pause
    exit /b 1
)

echo.
echo Step 1: Installing requirements...
pip install -r requirements.txt

echo.
echo Step 2: Quick test of core modules...
cd src
python -c "from nlp.language_detector import detect_language; print('Language detector OK:', detect_language('test query'))"
python -c "from kg.agri_kg import AgriculturalKG; kg=AgriculturalKG(); print('KG OK: nodes=', kg.G.number_of_nodes())"
cd ..

echo.
echo Step 3: Starting FastAPI backend (port 8000)...
start cmd /k "cd /d %~dp0 && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for API to start...
timeout /t 5 /nobreak

echo.
echo Step 4: Starting Streamlit frontend (port 8501)...
start cmd /k "cd /d %~dp0 && streamlit run frontend/app.py"

echo.
echo ============================================================
echo  System is starting!
echo  API:      http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo  Frontend: http://localhost:8501
echo ============================================================
pause
