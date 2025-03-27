@echo off
echo =======================================================
echo    Portable Image Classifier - Complete Build Process
echo =======================================================

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please install Python first.
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

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create directories if they don't exist
if not exist assets mkdir assets
if not exist sample_data mkdir sample_data

REM Generate sample data
echo Generating sample data...
python src/utils/generate_sample_data.py --num_samples 20 --img_size 64

REM Train model on sample data
echo Training model on sample data...
python src/train.py --data_dir sample_data --epochs 10 --batch_size 32

REM Build executable
echo Building executable...
python build_exe.py

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo =======================================================
echo Build process completed!
echo 
echo The executable is located in the 'dist' directory.
echo You can copy it to a USB drive and run it on any Windows computer.
echo =======================================================
pause 