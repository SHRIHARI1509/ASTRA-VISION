Write-Host "Starting Astra Vision backend API server on port 8000..."
.\venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
