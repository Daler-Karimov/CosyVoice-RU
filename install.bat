@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
title Установка CosyVoice-RU

echo ============================================================
echo   Установка CosyVoice-RU (Локальное клонирование голоса)
echo ============================================================
echo.

if not exist .venv (
    echo [1/3] Создание виртуального окружения Python...
    python -m venv .venv
) else (
    echo [1/3] Окружение .venv уже существует.
)

echo.
echo [2/3] Установка библиотек (PyTorch CUDA, Gradio, Whisper, CosyVoice)...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [3/3] Скачивание весов модели CosyVoice 3.0 (Fun-CosyVoice3-0.5B-2512)...
python download_cosyvoice.py

echo.
echo ============================================================
echo   Установка успешно завершена! Запустите run_ui.bat
echo ============================================================
pause
