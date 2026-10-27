@echo off
title RampSense Dashboard & Backend
echo ========================================================
echo Starting RampSense Backend Server & Web SCADA Dashboard
echo ========================================================
echo.

set PYTHON_EXE=C:\Users\archa\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo Using Python: %PYTHON_EXE%
echo Starting server on http://localhost:8000/
echo Press Ctrl+C in this console to stop the server anytime.
echo.

timeout /t 2 >nul
start "" "http://localhost:8000/"
"%PYTHON_EXE%" backend\server.py 8000

pause
