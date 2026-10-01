# ASTRA VISION — System Architecture (Phase 1)

## Overview
ASTRA VISION is an AI-based defence object recognition platform. During Phase 1 (Project Foundation), the architecture establishes decoupled layer boundaries before any machine learning capabilities or weight assets are introduced.

```
┌────────────────────────────────────────────────────────┐
│               Frontend (React + Vite)                  │
│       - Defence HUD Aesthetic & Telemetry Display      │
│       - Workspace Shell (Analysis Offline in Phase 1)  │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP (GET /api/health)
                            ▼
┌────────────────────────────────────────────────────────┐
│               Backend Gateway (FastAPI)                │
│       - API Routing (/api/health)                      │
│       - CORS Middleware & Settings Config              │
│       - Centralized Exception Handling & Logging       │
└───────────────────────────┬────────────────────────────┘
                            │ (Isolated Boundary)
                            ▼
┌────────────────────────────────────────────────────────┐
│             Model Services (Phase 2 Roadmap)           │
│       - Standby in Phase 1 (No weights/ML loaded)      │
│       - Clean architectural isolation guaranteed       │
└────────────────────────────────────────────────────────┘
```

## Layer Responsibilities
1. **Frontend (`frontend/`)**: React + TypeScript client providing a technical, responsive UI. Communicates with `/api/health` to monitor service telemetry.
2. **Backend Gateway (`backend/app/`)**: FastAPI application providing routing, configuration management, logging, CORS, and standardized JSON error envelopes.
3. **Model Services (`backend/app/services/`)**: Intentionally unpopulated in Phase 1. Strictly isolated from the API routing layer to prevent coupling.
