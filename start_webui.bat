@echo off
setlocal

REM AceStepAuto WebUI launcher (Gradio)
REM Uses ACE-Step embedded Python so dependencies are bundled.

set "PY=ACE-Step-1.5\python_embeded\python.exe"
set "ACESTEP_BAT=ACE-Step-1.5\start_api_server.bat"

if not exist "%PY%" (
  echo Embedded Python not found: %PY%
  echo Expected repo layout with ACE-Step-1.5\python_embeded\python.exe
  pause
  exit /b 1
)

if not exist "%ACESTEP_BAT%" (
  echo ACE-Step launcher not found: %ACESTEP_BAT%
  pause
  exit /b 1
)

REM Preferred port is 7865; if busy, webui.py auto-picks the next free port.
if "%WEBUI_PORT%"=="" set "WEBUI_PORT=7865"

REM Auto-open browser
if "%WEBUI_OPEN_BROWSER%"=="" set "WEBUI_OPEN_BROWSER=1"

echo Starting AceStepAuto WebUI (preferred port %WEBUI_PORT%)
echo It will print the final URL (may auto-increment if port is busy).
echo (Close this window to stop the WebUI)
echo.

REM If ACE-Step API is not running, start it in a separate window.
"%PY%" -c "import socket; s=socket.socket(); s.settimeout(0.5); ok=(s.connect_ex(('127.0.0.1',8001))==0); s.close(); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
  echo ACE-Step API not detected on 127.0.0.1:8001
  echo Launching %ACESTEP_BAT% ...
  start "ACE-Step API" "%ACESTEP_BAT%"
) else (
  echo ACE-Step API detected on 127.0.0.1:8001
)
echo.

"%PY%" -u webui.py

pause
