@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python rmbg_app.py
pause
