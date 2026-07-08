@echo off
chcp 65001 >nul
echo Building LegrestOptimizer one-file version...
pyinstaller --noconfirm --clean --onefile --windowed --name LegrestOptimizer --add-data "data;data" main.py
echo.
echo EXE path: dist\LegrestOptimizer.exe
pause
