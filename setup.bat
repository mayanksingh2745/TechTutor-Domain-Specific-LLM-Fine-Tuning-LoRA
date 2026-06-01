@echo off
echo ======================================================================
echo             TechTutor Environment Setup & Installation
echo ======================================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

:: Create Virtual Environment
if not exist .venv (
    echo [INFO] Creating virtual environment in .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment (.venv) already exists.
)

echo.
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [INFO] Checking for NVIDIA CUDA capability...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" >temp_cuda.txt 2>nul
set /p CUDA_STATUS=<temp_cuda.txt
del temp_cuda.txt

echo %CUDA_STATUS%
if "%CUDA_STATUS%"=="CUDA Available: True" (
    echo [INFO] NVIDIA CUDA detected. Installing CUDA-compatible PyTorch and dependencies...
    :: If they want to ensure GPU PyTorch, install it, else let requirements handle it
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [INFO] No CUDA GPU detected or CPU PyTorch is active.
    echo TechTutor will install requirements and run in CPU/Adaptive mode locally.
)

echo.
echo [INFO] Installing required dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed to install. Please check errors above.
) else (
    echo [SUCCESS] All dependencies installed successfully!
)

echo.
echo ======================================================================
echo Setup completed! To run the application:
echo 1. Activate the environment:   call .venv\Scripts\activate
echo 2. Generate the dataset:      python src/dataset_generator.py
echo 3. Run the interactive dashboard: streamlit run app.py
echo ======================================================================
echo.
pause
