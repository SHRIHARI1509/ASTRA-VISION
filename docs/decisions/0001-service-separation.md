# ADR 0001: Separation of Frontend, Backend, and Model Services

## Status
Accepted

## Context
ASTRA VISION is an AI-based defence object recognition platform. Coupling model inference directly into web request handlers leads to monolithic lock-in, resource contention, and testing complexities. 

Phase 1 requires establishing a verified full-stack project foundation before introducing machine learning libraries or models.

## Decision
Enforce strict separation between:
1. `frontend/`: Reactive UI and telemetry display.
2. `backend/app/api/`: Routing, request validation, and error envelope handling.
3. `backend/app/services/`: Reserved for model lifecycle and ML inference in Phase 2. Zero ML code or dependencies in Phase 1.
4. `tests/`: Independent test suites for unit, contract, and cross-layer integration.

## Consequences
- Clean, fast local development environment with minimal dependencies.
- Frontend and backend can be tested and verified independently.
- Avoids premature dependency bloat (PyTorch, Transformers, YOLO, OpenCV excluded in Phase 1).
