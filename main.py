@echo off
chcp 65001 >nul
echo Building LegrestOptimizer folder version...
pyinstaller --noconfirm --clean --windowed --name LegrestOptimizer --add-data "data;data" main.py
echo.
echo EXE path: dist\LegrestOptimizer\LegrestOptimizer.exe
pause
