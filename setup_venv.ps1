# setup_venv.ps1 - Venv Setup for 3D Segmentation Task

Write-Host "--- Starting Environment Setup ---" -ForegroundColor Cyan

# 1. Clean up potential "junk" from failed installations (pip cache and temp files)
Write-Host "Cleaning up pip cache to free space..." -ForegroundColor Yellow
python -m pip cache purge

# 2. Create the Virtual Environment
$VENV_DIR = "segmentation_venv"
if (!(Test-Path $VENV_DIR)) {
    Write-Host "Creating Virtual Environment in $VENV_DIR..." -ForegroundColor Green
    python -m venv $VENV_DIR
}
else {
    Write-Host "Virtual Environment already exists." -ForegroundColor Yellow
}

# 3. Activation Command
$ACTIVATE = "$VENV_DIR\Scripts\Activate.ps1"
Write-Host "Activating Venv..." -ForegroundColor Green
& $ACTIVATE

# 4. Install Dependencies with space-saving flags
Write-Host "Installing dependencies (this may take time)..." -ForegroundColor Green
# Using --no-cache-dir to minimize disk usage during installation
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir monai[all] pyvista nibabel scikit-image rtree matplotlib

Write-Host "--- Setup Complete ---" -ForegroundColor Cyan
Write-Host "To run your script, use: python medical_segmentation_v1.py"
