@echo off
SETLOCAL EnableDelayedExpansion

:: FluxRT installation script for Windows.
:: Run from the repository root: scripts\install.bat

:: ── sanity-check: running from repo root ──────────────────────────────────────
IF NOT EXIST "pyproject.toml" (
    echo [ERROR] This script must be run from the FluxRT repository root.
    exit /b 1
)

:: ── prerequisites ─────────────────────────────────────────────────────────────
echo [+] Checking prerequisites...

where git >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] 'git' is not installed. Install it from https://git-scm.com/download/win
    exit /b 1
)

SET UV_DIR=thirdparty
SET UV=%UV_DIR%\uv.exe

IF EXIST "%UV%" (
    echo [+] uv already present at '%UV%'.
) ELSE (
    echo [!] uv not found. Downloading to '%UV_DIR%'...
    IF NOT EXIST "%UV_DIR%" mkdir "%UV_DIR%"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$zip = [System.IO.Path]::GetTempFileName() + '.zip';" ^
        "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip;" ^
        "Expand-Archive $zip -DestinationPath '%UV_DIR%' -Force;" ^
        "Remove-Item $zip"
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to download uv. Check your internet connection and re-run.
        exit /b 1
    )
    IF NOT EXIST "%UV%" (
        echo [ERROR] uv.exe not found after extraction. Re-run the script.
        exit /b 1
    )
    echo [+] uv installed successfully to '%UV_DIR%'.
)

git lfs version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] 'git-lfs' is not installed.
    echo         Install with: winget install GitHub.GitLFS
    echo         Or download from https://git-lfs.com
    exit /b 1
)

echo [+] All prerequisites found.

:: ── uv virtual environment ───────────────────────────────────────────────────
SET VENV_DIR=fluxrt

IF EXIST "%VENV_DIR%" (
    echo [+] Virtual environment '%VENV_DIR%' already exists.
) ELSE (
    echo [+] Creating virtual environment '%VENV_DIR%' with python=3.12 via uv...
    "%UV%" venv "%VENV_DIR%" --python 3.12
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment via uv.
        exit /b 1
    )
)

CALL "%VENV_DIR%\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate virtual environment '%VENV_DIR%'.
    exit /b 1
)

:: ── PyTorch ───────────────────────────────────────────────────────────────────
python -c "import torch" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing PyTorch with XPU support...
    "%UV%" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install PyTorch.
        exit /b 1
    )
) ELSE (
    echo [+] PyTorch is already installed.
)

:: ── Python requirements ───────────────────────────────────────────────────────
:: Use 'diffusers' as a proxy — it's the heaviest transitive dependency.
python -c "import diffusers" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing Python requirements from requirements.txt...
    "%UV%" pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install requirements.
        exit /b 1
    )
) ELSE (
    echo [+] Python requirements already installed.
)

:: ── triton-windows ────────────────────────────────────────────────────────────
:: Required for model compilation on Windows (auto-installed but explicit here).
python -c "import triton" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing triton-windows (required for model compilation^)...
    "%UV%" pip install triton-windows
    IF ERRORLEVEL 1 (
        echo [!] Warning: triton-windows installation failed.
        echo [!]          Model compilation may not work. Check compatibility at:
        echo [!]          https://github.com/woct0rdho/triton-windows/issues/158
    )
) ELSE (
    echo [+] triton already installed.
)

:: ── fluxrt package ────────────────────────────────────────────────────────────
python -c "from fluxrt import StreamProcessor" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing fluxrt package in editable mode...
    "%UV%" pip install -e .
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install fluxrt package.
        exit /b 1
    )
) ELSE (
    echo [+] fluxrt package already installed.
)

:: ── model downloads ───────────────────────────────────────────────────────────
:: Register LFS hooks for the current user (idempotent).
git lfs install

:: ── RIFE frame-interpolation model ───────────────────────────────────────────
SET RIFE_DIR=RIFE-safetensors
SET RIFE_SENTINEL=RIFE-safetensors\flownet.safetensors
IF EXIST "%RIFE_SENTINEL%" (
    echo [+] RIFE frame-interpolation model: already downloaded.
) ELSE IF EXIST "%RIFE_DIR%\.git" (
    echo [!] RIFE: directory exists but looks incomplete — resuming LFS download...
    git -C "%RIFE_DIR%" pull --ff-only
    git -C "%RIFE_DIR%" lfs pull
) ELSE IF EXIST "%RIFE_DIR%" (
    echo [!] RIFE: '%RIFE_DIR%' exists but is not a git repository.
    echo [!]       Remove it and re-run to download the model.
) ELSE (
    echo [+] Downloading RIFE frame-interpolation model...
    git clone https://huggingface.co/TensorForger/RIFE-safetensors
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to clone RIFE model.
        exit /b 1
    )
)

:: ── FLUX.2-klein-4B base model ────────────────────────────────────────────────
SET FLUX_DIR=FLUX.2-klein-4B
SET FLUX_SENTINEL=FLUX.2-klein-4B\transformer\diffusion_pytorch_model.safetensors
IF EXIST "%FLUX_SENTINEL%" (
    echo [+] FLUX.2-klein-4B base model: already downloaded.
) ELSE IF EXIST "%FLUX_DIR%\.git" (
    echo [!] FLUX.2-klein-4B: directory exists but looks incomplete — resuming LFS download...
    git -C "%FLUX_DIR%" pull --ff-only
    git -C "%FLUX_DIR%" lfs pull
) ELSE IF EXIST "%FLUX_DIR%" (
    echo [!] FLUX.2-klein-4B: '%FLUX_DIR%' exists but is not a git repository.
    echo [!]                  Remove it and re-run to download the model.
) ELSE (
    echo [+] Downloading FLUX.2-klein-4B base model...
    git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to clone FLUX.2-klein-4B model.
        exit /b 1
    )
)

:: ── FLUX.2-klein-4B-int8 model ────────────────────────────────────────────────
SET INT8_DIR=FLUX.2-klein-4B-int8
SET INT8_SENTINEL=FLUX.2-klein-4B-int8\diffusion_pytorch_model.safetensors
IF EXIST "%INT8_SENTINEL%" (
    echo [+] FLUX.2-klein-4B int8 model: already downloaded.
) ELSE IF EXIST "%INT8_DIR%\.git" (
    echo [!] FLUX.2-klein-4B-int8: directory exists but looks incomplete — resuming LFS download...
    git -C "%INT8_DIR%" pull --ff-only
    git -C "%INT8_DIR%" lfs pull
) ELSE IF EXIST "%INT8_DIR%" (
    echo [!] FLUX.2-klein-4B-int8: '%INT8_DIR%' exists but is not a git repository.
    echo [!]                       Remove it and re-run to download the model.
) ELSE (
    echo [+] Downloading FLUX.2-klein-4B int8 model...
    git clone https://huggingface.co/aydin99/FLUX.2-klein-4B-int8
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to clone FLUX.2-klein-4B-int8 model.
        exit /b 1
    )
)

:: ── done ──────────────────────────────────────────────────────────────────────
echo.
echo [+] Installation complete.
echo [!] Note: the GUI requires OBS to be installed for virtual webcam output.
echo [!]       Download from https://obsproject.com/download
echo.
echo [+] Activate the environment and start:  .\%VENV_DIR%\Scripts\activate
echo [+] Then run, for example:               python scripts\run_gradio_demo.py

ENDLOCAL
