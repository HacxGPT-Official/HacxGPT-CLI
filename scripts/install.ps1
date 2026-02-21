# HacxGPT Installer for Windows (PowerShell)
# https://github.com/BlackTechX011/Hacx-GPT

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "    HacxGPT Installer for Windows" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 1. Check for Python
Write-Host "[~] Checking for Python..." -ForegroundColor Cyan
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "[!] Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    pause
    exit
}
Write-Host "[+] Python found." -ForegroundColor Green

# 2. Check for Git
Write-Host "[~] Checking for Git..." -ForegroundColor Cyan
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Git is not installed. We will attempt to download the source directly..." -ForegroundColor Yellow
} else {
    Write-Host "[+] Git found." -ForegroundColor Green
}

# 3. Handle Source Code
if (!(Test-Path "setup.py")) {
    Write-Host "[~] Downloading HacxGPT source code..." -ForegroundColor Cyan
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git clone https://github.com/HacxGPT-Official/HacxGPT-CLI.git
        Set-Location HacxGPT-CLI
    } else {
        Write-Host "[!] Git missing. Please clone the repository manually: https://github.com/HacxGPT-Official/HacxGPT-CLI" -ForegroundColor Red
        pause
        exit
    }
}

# 4. Create Virtual Environment
Write-Host "[~] Creating Virtual Environment (.venv)..." -ForegroundColor Cyan
if (!(Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "[+] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[~] Virtual environment already exists." -ForegroundColor Gray
}

# 5. Install Dependencies
Write-Host "[~] Installing HacxGPT and dependencies..." -ForegroundColor Cyan
try {
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    & .venv\Scripts\python.exe -m pip install -e .
    Write-Host "[+] Installation successful." -ForegroundColor Green
}
catch {
    Write-Host "[!] Installation failed: $_" -ForegroundColor Red
    pause
    exit
}

Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "      Installation Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "To run HacxGPT:" -ForegroundColor Yellow
Write-Host "1. .venv\Scripts\activate"
Write-Host "2. hacxgpt"
Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
