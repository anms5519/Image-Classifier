@echo off
echo ===================================================
echo    Portable Image Classifier - Starting Application
echo ===================================================

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please install Python first or use the standalone executable.
    pause
    exit /b
)

REM Check if venv exists, create if it doesn't
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if packages are installed
python -c "import torch" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run the application
python src/main.py

REM Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat

echo Application closed.
pause 