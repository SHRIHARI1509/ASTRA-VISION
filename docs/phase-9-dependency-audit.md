# Phase 9: Dependency & Configuration Audit

**Document ID:** `AV-DEP-09H`  
**Phase:** 9.8 Dependency / Configuration Audit  
**Date:** October 2026  
**Auditor:** Final Release Engineer  
**Status:** PASS (STABLE & CONSERVATIVE)  

---

## 1. Executive Summary

A comprehensive dependency and environment configuration audit was performed across both the Python backend and Node/TypeScript frontend. All declarations, lockfiles, version boundaries, build setups, and runtime assumptions were analyzed.

* **Backend Requirements:** Standard, minimal, production-focused ([`backend/requirements.txt`](file:///c:/FILES/astra-vision/backend/requirements.txt))
* **Frontend Packages:** Modern React 19 + Vite 8 stack ([`frontend/package.json`](file:///c:/FILES/astra-vision/frontend/package.json))
* **Hardcoded Local Paths in Core Code:** **0** (All relative paths and dynamic directory resolvers)
* **API Proxy Alignment:** Frontend Vite proxy matches backend default port (`8000`)
* **Dependency Health:** **STABLE & VERIFIED (PASS)**

---

## 2. Backend Dependencies Audit

File: [`backend/requirements.txt`](file:///c:/FILES/astra-vision/backend/requirements.txt)

| Package | Declared Version | Purpose | Necessity Assessment |
| :--- | :--- | :--- | :--- |
| `fastapi` | `>=0.110.0` | Core REST API web framework | Essential production dependency |
| `uvicorn[standard]` | `>=0.28.0` | High-performance ASGI server | Essential production dependency |
| `pydantic` | `>=2.6.0` | Request/response data validation | Essential production dependency |
| `pydantic-settings` | `>=2.2.0` | Environment settings management | Essential production dependency |
| `python-dotenv` | `>=1.0.0` | `.env` configuration file loader | Essential production dependency |
| `pytest` | `>=8.0.0` | Automated regression test framework | Development/Test dependency |
| `httpx` | `>=0.27.0` | Async HTTP client for ASGI testing | Test dependency |
| `pillow` | `>=10.2.0` | Image parsing, format verification | Essential production dependency |
| `python-multipart` | `>=0.0.9` | Multipart form-data parser for uploads | Essential production dependency |
| `torch` | `>=2.0.0` | PyTorch execution runtime | Essential model inference dependency |
| `transformers` | `>=4.40.0` | Hugging Face SigLIP 2 architecture | Essential model inference dependency |

### Notes on Optional / Experimental Dependencies
* `peft`: Installed in local virtualenv (`0.21.1`) and used exclusively by offline fine-tuning modules ([`backend/app/finetuning/trainer.py`](file:///c:/FILES/astra-vision/backend/app/finetuning/trainer.py) and [`scripts/run_phase8g_finetuning.py`](file:///c:/FILES/astra-vision/scripts/run_phase8g_finetuning.py)). It is intentionally omitted from the core runtime `requirements.txt` to keep the production inference image lightweight.

---

## 3. Frontend Dependencies Audit

File: [`frontend/package.json`](file:///c:/FILES/astra-vision/frontend/package.json)

| Package | Declared Version | Classification | Assessment |
| :--- | :--- | :--- | :--- |
| `react` | `^19.2.8` | Runtime Dependency | Modern React UI engine |
| `react-dom` | `^19.2.8` | Runtime Dependency | React DOM renderer |
| `@testing-library/react` | `^16.3.3` | devDependency | Component unit testing |
| `@types/node` | `^24.13.3` | devDependency | TypeScript Node types |
| `@types/react` | `^19.2.18` | devDependency | TypeScript React definitions |
| `@types/react-dom` | `^19.2.7` | devDependency | TypeScript React DOM definitions |
| `@vitejs/plugin-react` | `^6.1.1` | devDependency | Vite React Fast Refresh plugin |
| `jsdom` | `^29.1.1` | devDependency | DOM simulation for Vitest |
| `oxlint` | `^1.81.0` | devDependency | Fast Rust-based linter |
| `typescript` | `~6.0.2` | devDependency | Type checker |
| `vite` | `^8.3.0` | devDependency | Bundler and dev server |
| `vitest` | `^5.0.2` | devDependency | Unit test runner |

* **Finding:** Zero bloat. Only 2 production dependencies (`react`, `react-dom`). All dev dependencies are appropriate for testing and compiling.

---

## 4. Environment & Network Assumptions

1. **Proxy Alignment:** [`frontend/vite.config.ts`](file:///c:/FILES/astra-vision/frontend/vite.config.ts) proxies `/api` requests to `http://127.0.0.1:8000`, cleanly matching the backend defaults in [`backend/app/core/config.py`](file:///c:/FILES/astra-vision/backend/app/core/config.py).
2. **CORS Alignment:** Backend `ALLOWED_ORIGINS` explicitly includes `http://localhost:5173` and `http://127.0.0.1:5173`.
3. **Device Fallback:** Model inference adapter defaults to `auto`, seamlessly selecting CUDA if present or falling back to CPU with full error handling.

---

## 5. Audit Verdict

| Audit Item | Result | Status |
| :--- | :--- | :---: |
| Dependency Bloat | Clean, minimal dependencies in both stacks | **PASS** |
| Hardcoded Local Paths | Zero hardcoded paths in production code | **PASS** |
| Config Mismatches | Frontend proxy and backend host/port match | **PASS** |
| Type Safety | Strict TypeScript typecheck passing | **PASS** |
| Build Stability | Vite production build passing | **PASS** |
