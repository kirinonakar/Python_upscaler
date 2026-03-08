@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start /b pythonw ctk_app.py
timeout /t 3
exit
