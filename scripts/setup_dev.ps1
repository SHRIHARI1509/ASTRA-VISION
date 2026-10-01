Write-Host "Setting up Astra Vision development environment..."

# Setup backend
Write-Host "Setting up Python virtual environment..."
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

# Setup frontend
Write-Host "Installing frontend dependencies..."
cd frontend
npm install
cd ..

Write-Host "Development environment setup complete!"
