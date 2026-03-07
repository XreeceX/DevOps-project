@echo off
REM Regenerate visualization images and copy to showcase
echo Installing Python deps...
pip install -r requirements.txt -q
echo Running fraud detection pipeline...
python src/Fraud_Detection.py
if %ERRORLEVEL% neq 0 (
  echo Pipeline failed. Make sure you have Python with pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, openpyxl.
  exit /b 1
)
echo Copying images to showcase/public...
if not exist "showcase\public" mkdir showcase\public
copy /Y static\*.png showcase\public\
echo Done. Commit showcase/public/*.png and push.
pause
