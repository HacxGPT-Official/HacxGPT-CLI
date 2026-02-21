# HacxGPT Update Script for Windows (PowerShell)
# https://github.com/BlackTechX011/Hacx-GPT

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "    HacxGPT System Updater (PS)" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 1. Check for Git
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Error: Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "[!] Please install Git from https://git-scm.com/" -ForegroundColor Yellow
    exit
}

# 2. Check if .git exists
if (!(Test-Path ".git")) {
    Write-Host "[!] Error: Not a git repository. Please clone from GitHub properly." -ForegroundColor Red
    exit
}

# 3. Pull from GitHub
Write-Host "[~] Fetching latest changes from main branch..." -ForegroundColor Cyan
try {
    git pull origin main
    Write-Host "[+] Codebase synchronized." -ForegroundColor Green
}
catch {
    Write-Host "[!] Error during git pull: $_" -ForegroundColor Red
    exit
}

# 4. Update dependencies
Write-Host "[~] Updating dependencies..." -ForegroundColor Cyan
try {
    # Check for virtual environment
    $pythonExe = "python"
    if (Test-Path ".venv\Scripts\python.exe") {
        $pythonExe = ".venv\Scripts\python.exe"
        Write-Host "[+] Using virtual environment: .venv" -ForegroundColor Gray
    }

    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -e .
    Write-Host "[+] Dependencies updated successfully." -ForegroundColor Green
}
catch {
    Write-Host "[!] Error during dependency update: $_" -ForegroundColor Red
    exit
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "      Update Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Restart HacxGPT to apply the latest changes." -ForegroundColor Yellow
pause
