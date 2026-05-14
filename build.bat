@echo off
echo Building Vibration Analyser...
pip install -r requirements.txt
pyinstaller app.spec --clean
echo.
echo Build complete. Distribute the folder: dist\VibrationAnalyser\
echo Run: dist\VibrationAnalyser\VibrationAnalyser.exe
pause
