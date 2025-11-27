# backend/api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from .emit_ui_payload import build_payload
import json
import traceback

app = FastAPI()

# Allow your Netlify frontend + local dev
origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    # TODO: replace with your actual Netlify URL, e.g.:
    # "https://stockpredictor-ui.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    # In dev, this is fine; for production prefer the explicit `origins` list above.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default values matching ui_config.json
DEFAULT_START = "2000-01-01"
DEFAULT_MODELS_DIR = str(
    Path(__file__).resolve().parents[1] / "app" / "models"
)


@app.get("/api/predict")
def predict(
    ticker: str = Query(...),
    model: str = Query("xgb"),
):
    ticker = ticker.upper()
    try:
        payload = build_payload(
            ticker=ticker,
            start=DEFAULT_START,
            models_dir=DEFAULT_MODELS_DIR,
            model=model,
        )

        # Optionally mirror desktop behavior and write JSON to disk
        out_path = Path(DEFAULT_MODELS_DIR) / f"ui_payload_{ticker}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return payload

    except SystemExit as e:
        # Expected "user-level" errors (e.g., not enough data)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Log full traceback to server logs for debugging
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
