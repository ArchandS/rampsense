# RampSense Dashboard Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting RampSense Backend Server & Web SCADA Dashboard" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$pythonExe = "C:\Users\archa\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Yellow
Write-Host "Opening Dashboard at http://localhost:8000/" -ForegroundColor Green

Start-Process "http://localhost:8000/"
& $pythonExe "backend\server.py" 8000
