"""
Convenience entry point: `python main.py` starts the dev server.

Equivalent to `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
Must be run from the repo root (so the `app` package is importable).
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
