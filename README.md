# ⚡ Freetown Power — Load Shedding Risk Predictor

AI-powered 24-hour electricity outage risk forecast for Freetown, Sierra Leone.

Built on real EDSA (Electricity Distribution and Supply Authority) hourly operational data, 2022–2025.

---

## What It Does

- Predicts **system-level load shedding risk** for the next 24 hours
- Color-coded risk levels: 🟢 Safe · 🟡 Stressed · 🔴 Likely shedding
- Shows current supply vs demand gap in MW
- Powered by a CNN-BiLSTM deep learning model trained on 17,532 hourly EDSA records

## Stack

- **Backend:** Python / FastAPI
- **Model:** CNN-BiLSTM + Multi-Head Attention (TensorFlow 2.16)
- **Frontend:** HTML / Tailwind CSS / Chart.js
- **Data:** EDSA hourly load data (utilized_mw, available_mw)

## Risk Thresholds

| Util Ratio | Risk Level | Meaning |
|-----------|-----------|---------|
| < 0.80 | 🟢 Safe | Supply comfortable |
| 0.80 – 0.92 | 🟡 Stressed | Reduced margins |
| > 0.92 | 🔴 Critical | Load shedding likely |

## Project Structure

```
freetown-power/
├── app/
│   ├── main.py          # FastAPI app
│   ├── predictor.py     # Model inference + risk scoring
│   └── data_loader.py   # EDSA data pipeline
├── model/               # Trained CNN-BiLSTM weights
├── data/                # Processed EDSA hourly data
├── templates/           # HTML dashboard
├── static/              # CSS + JS
└── requirements.txt
```

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# Open http://localhost:8000
```

---

Built by Al-0991 · Data: EDSA Sierra Leone · Model: CNN-BiLSTM v3
