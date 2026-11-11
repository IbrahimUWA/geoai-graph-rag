@echo off
rem Start GraphRAG Explorer (opens http://127.0.0.1:7861 in your browser).
cd /d "%~dp0"
call .venv\Scriptsctivate.bat
python -m graphrag.app
pause
