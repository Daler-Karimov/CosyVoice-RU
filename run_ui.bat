@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
title CosyVoice 3.0 WebUI
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" webui.py
pause
