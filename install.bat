@echo off
rem GraphRAG Explorer - one-step installer for Windows 10/11.
rem   Double-click install.bat (or run it in a terminal).
rem Creates .venv, installs PyTorch (GPU if an NVIDIA card is found, otherwise CPU), the app's packages,
rem the ready-built knowledge graph, a local language model through Ollama, and a desktop shortcut.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PY=
where py >nul 2>nul && (for %%v in (3.12 3.11 3.13 3.10) do if not defined PY (py -%%v -c "1" >nul 2>nul && set "PY=py -%%v"))
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python 3.10-3.13 is needed. Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this again.
  pause & exit /b 1
)

echo.
echo ==^> Creating the virtual environment (.venv) with %PY%
%PY% -m venv .venv || (echo Could not create the virtual environment & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q

nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo ==^> NVIDIA GPU found: installing PyTorch with CUDA
  pip install -q torch --index-url https://download.pytorch.org/whl/cu128
) else (
  echo ==^> No NVIDIA GPU: installing the CPU build of PyTorch ^(the app works fine on CPU^)
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
)

echo ==^> Installing the app's packages
pip install -q -r requirements.txt || (echo Package install failed & pause & exit /b 1)

echo ==^> Downloading the ready-built knowledge graph
python scripts\get_data.py || (echo Download failed & pause & exit /b 1)

echo ==^> Language model (Ollama)
where ollama >nul 2>nul
if errorlevel 1 (
  choice /c YN /m "Ollama is not installed. Install it now with winget"
  if !errorlevel!==1 winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
  set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
)
where ollama >nul 2>nul
if errorlevel 1 (
  echo Skipped. Install Ollama later from https://ollama.com and run: ollama pull qwen3:8b
) else (
  for /f %%m in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"') do set MEM=%%m
  set MODEL=qwen3:8b
  if !MEM! LSS 16 set MODEL=qwen3:4b
  echo Pulling !MODEL! ^(about 5 GB the first time^)
  ollama pull !MODEL!
  > settings.json echo {"llm_backend": "ollama", "llm_model": "!MODEL!"}
)

echo ==^> Adding a desktop shortcut
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\GraphRAG Explorer.lnk'); $s.TargetPath='%~dp0run_app.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0graphrag\assets\icon.ico'; $s.Save()"

echo.
echo Done. Start the app with the "GraphRAG Explorer" shortcut on your desktop, or run_app.bat
pause
