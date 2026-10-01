# Testing Guide — Phase 1

This guide covers running unit, contract, and build verification tests for Phase 1.

## 1. Backend Unit Tests
Executes unit tests verifying route responses, status codes, and error envelopes.

```powershell
.\venv\Scripts\pytest backend\tests
```

## 2. Integration & Contract Tests
Verifies that the API conforms to the expected contract (`/api/health` returning `{"status": "ok", "service": "astra-vision"}`).

```powershell
$env:PYTHONPATH="."
.\venv\Scripts\pytest tests\integration
```

## 3. Frontend Typecheck and Build Verification
Verifies that TypeScript compiles without errors and the Vite production bundle builds successfully.

```powershell
cd frontend
npm test        # Runs tsc -b type verification
npm run build   # Produces production distribution in dist/
```
