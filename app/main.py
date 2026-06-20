"""
main.py
FastAPI app — Freetown Power load shedding risk dashboard.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, json
from datetime import datetime

from app.data_loader import load_data, fit_scaler, get_latest_window, get_risk_context
from app.predictor import forecast_24h, summary_risk

app = FastAPI(title="Freetown Power", description="Load shedding risk predictor for Freetown, Sierra Leone")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load data and model once at startup
_df = None
_scaler = None

@app.on_event("startup")
async def startup():
    global _df, _scaler
    _df = load_data()
    _scaler = fit_scaler(_df)
    print(f"✅ EDSA data loaded: {len(_df):,} records")
    print(f"✅ Scaler fitted on 2022-2023 training data")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    window = get_latest_window(_df, _scaler)
    last_available = float(_df['available_mw'].iloc[-1])
    last_utilized  = float(_df['utilized_mw'].iloc[-1])
    last_timestamp = str(_df['timestamp'].iloc[-1])

    forecast = forecast_24h(window, _scaler, last_available)
    summary  = summary_risk(forecast)
    history  = get_risk_context(_df)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "summary": summary,
        "forecast": forecast,
        "history": history,
        "last_timestamp": last_timestamp,
        "last_utilized": round(last_utilized, 1),
        "last_available": round(last_available, 1),
        "current_ratio": round(last_utilized / max(last_available, 1), 3),
        "forecast_json": json.dumps(forecast),
        "history_json": json.dumps(history),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })


@app.get("/api/forecast")
async def api_forecast():
    window = get_latest_window(_df, _scaler)
    last_available = float(_df['available_mw'].iloc[-1])
    forecast = forecast_24h(window, _scaler, last_available)
    summary  = summary_risk(forecast)
    return {"summary": summary, "forecast": forecast}


@app.get("/api/status")
async def api_status():
    return {
        "status": "online",
        "records_loaded": len(_df),
        "latest_data": str(_df['timestamp'].iloc[-1]),
        "model": "CNN-BiLSTM v3 with Multi-Head Attention",
    }
