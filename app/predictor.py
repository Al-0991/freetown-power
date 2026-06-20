"""
predictor.py
Loads trained CNN-BiLSTM v3 model and generates 24-hour risk forecast.
"""
import numpy as np
import tensorflow as tf
import os
from app.data_loader import classify_risk, risk_label, risk_color, FEATURES

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_v3.keras")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def inverse_target(scaled_vals, scaler):
    """Inverse-transform only the target column (utilized_mw)."""
    target_col = FEATURES.index('utilized_mw')
    n = scaled_vals.shape[0] * scaled_vals.shape[1]
    dummy = np.zeros((n, len(FEATURES)))
    dummy[:, target_col] = scaled_vals.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col].reshape(scaled_vals.shape)


def forecast_24h(window: np.ndarray, scaler, last_available_mw: float):
    """
    Run model inference on the latest 168-hour window.
    Returns a list of 24 hourly forecast dicts.
    """
    model = get_model()
    pred_scaled = model.predict(window, verbose=0)          # (1, 24)
    pred_mw = inverse_target(pred_scaled, scaler)[0]        # (24,)

    results = []
    for h, mw in enumerate(pred_mw):
        mw = float(np.clip(mw, 20, 200))
        util_ratio = mw / max(last_available_mw, 1.0)
        level = classify_risk(util_ratio)
        results.append({
            'hour_offset': h + 1,
            'predicted_mw': round(mw, 1),
            'available_mw': round(last_available_mw, 1),
            'deficit_mw': round(max(0, mw - last_available_mw), 1),
            'util_ratio': round(util_ratio, 3),
            'risk_level': level,
            'risk_label': risk_label(level),
            'risk_color': risk_color(level),
        })
    return results


def summary_risk(forecast: list) -> dict:
    """Aggregate 24-hour forecast into a top-level risk summary."""
    levels = [f['risk_level'] for f in forecast]
    critical_hours = levels.count('critical')
    stressed_hours = levels.count('stressed')
    peak_ratio = max(f['util_ratio'] for f in forecast)
    peak_hour = next(f['hour_offset'] for f in forecast if f['util_ratio'] == peak_ratio)

    if critical_hours >= 3:
        overall = 'critical'
    elif critical_hours >= 1 or stressed_hours >= 6:
        overall = 'stressed'
    else:
        overall = 'safe'

    return {
        'overall_risk': overall,
        'overall_label': risk_label(overall),
        'overall_color': risk_color(overall),
        'critical_hours': critical_hours,
        'stressed_hours': stressed_hours,
        'peak_util_ratio': round(peak_ratio, 3),
        'peak_hour_offset': peak_hour,
        'advice': _advice(overall, critical_hours, peak_ratio),
    }


def _advice(level, critical_hours, peak_ratio):
    if level == 'critical':
        return f"High load shedding risk. {critical_hours} critical hour(s) forecast. Charge devices and backup power now."
    elif level == 'stressed':
        return f"Grid under stress. Peak demand reaches {round(peak_ratio*100)}% of available supply. Prepare for possible interruptions."
    else:
        return "Grid supply is comfortable for the next 24 hours. No significant shedding expected."
