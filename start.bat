@echo off
title BigBrother
cd /d "%~dp0"

echo.
echo  Starting BigBrother...
echo  The dashboard opens in your browser. Press Ctrl+C here to stop.
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    python main.py %*
)

pause
