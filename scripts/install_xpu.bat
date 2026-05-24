@echo off
SETLOCAL EnableDelayedExpansion

:: FluxRT XPU installation script for Windows (Intel GPU).
:: Run from the repository root: scripts\install_xpu.bat

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

git lfs version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] 'git-lfs' is not installed.
    echo         Install with: winget install GitHub.GitLFS
    echo         Or download from https://git-lfs.com
    exit /b 1
)

:: ── uv ────────────────────────────────────────────────────────────────────────
where uv >nul 2>&1
IF ERRORLEVEL 1 (
    echo [!] 'uv' not found. Installing via official binary installer...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install uv. Install it manually: https://docs.astral.sh/uv/
        exit /b 1
    )
    :: Reload PATH so uv is visible in this session
    FOR /F "delims=" %%i IN ('powershell -Command "[System.Environment]::GetEnvironmentVariable(\"PATH\",\"User\")"') DO SET "PATH=%%i;%PATH%"
    where uv >nul 2>&1
    IF ERRORLEVEL 1 (
        echo [ERROR] uv was installed but is not on PATH. Open a new terminal and re-run.
        exit /b 1
    )
)

echo [+] All prerequisites found.

:: ── virtual environment ───────────────────────────────────────────────────────
SET VENV_DIR=fluxrt

IF EXIST "%VENV_DIR%\Scripts\activate.bat" (
    echo [+] Virtual environment '%VENV_DIR%' already exists.
) ELSE (
    echo [+] Creating virtual environment '%VENV_DIR%' (python=3.12^) via uv...
    uv venv "%VENV_DIR%" --python 3.12
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

CALL "%VENV_DIR%\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate virtual environment '%VENV_DIR%'.
    exit /b 1
)

:: ── PyTorch with XPU support ──────────────────────────────────────────────────
python -c "import torch" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing PyTorch with Intel XPU support...
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install PyTorch.
        exit /b 1
    )
) ELSE (
    echo [+] PyTorch is already installed.
)

:: ── Python requirements ───────────────────────────────────────────────────────
python -c "import diffusers" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing Python requirements from requirements.txt...
    uv pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install requirements.
        exit /b 1
    )
) ELSE (
    echo [+] Python requirements already installed.
)

:: ── fluxrt package ────────────────────────────────────────────────────────────
python -c "import fluxrt" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [+] Installing fluxrt package in editable mode...
    uv pip install -e .
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install fluxrt package.
        exit /b 1
    )
) ELSE (
    echo [+] fluxrt package already installed.
)

:: ── model downloads ───────────────────────────────────────────────────────────
git lfs install

:: ── RIFE frame-interpolation model ───────────────────────────────────────────
SET RIFE_DIR=RIFE-safetensors
SET RIFE_SENTINEL=RIFE-safetensors\flownet.safetensors
IF EXIST "%RIFE_SENTINEL%" (
    echo [+] RIFE frame-interpolation model: already downloaded.
) ELSE IF EXIST "%RIFE_DIR%\.git" (
    echo [!] RIFE: directory exists but looks incomplete -- resuming LFS download...
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
    echo [!] FLUX.2-klein-4B: directory exists but looks incomplete -- resuming LFS download...
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
    echo [!] FLUX.2-klein-4B-int8: directory exists but looks incomplete -- resuming LFS download...
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
echo [!] Note: set "device": "xpu" in your config JSON to use the Intel GPU.
echo [!] Note: the GUI requires OBS to be installed for virtual webcam output.
echo [!]       Download from https://obsproject.com/download
echo.
echo [+] Activate the environment and start:  %VENV_DIR%\Scripts\activate
echo [+] Then run, for example:               python scripts\run_gradio_demo.py
