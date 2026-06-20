"""
data_loader.py
Loads and prepares EDSA hourly data for inference.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hourly_clean.csv")

FEATURES = [
    'utilized_mw', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'month_sin', 'month_cos', 'is_weekend', 'lag_24h', 'lag_168h',
    'roll_24h_mean', 'util_ratio'
]
SEQ_LEN = 168


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df[df['is_gap'] == 0][FEATURES + ['timestamp', 'available_mw']].dropna()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def fit_scaler(df):
    train_end = df[df['timestamp'].dt.year <= 2023].index[-1]
    scaler = MinMaxScaler()
    scaler.fit(df[FEATURES].iloc[:train_end])
    return scaler


def get_latest_window(df, scaler):
    """Return the most recent SEQ_LEN hours as a model-ready array."""
    recent = df.tail(SEQ_LEN).copy()
    scaled = scaler.transform(recent[FEATURES])
    return scaled.reshape(1, SEQ_LEN, len(FEATURES))


def get_risk_context(df):
    """Return last 48 hours of actual data for dashboard display."""
    recent = df.tail(48)[['timestamp', 'utilized_mw', 'available_mw']].copy()
    recent['util_ratio'] = recent['utilized_mw'] / recent['available_mw'].clip(lower=1)
    recent['risk'] = recent['util_ratio'].apply(classify_risk)
    return recent.to_dict(orient='records')


def classify_risk(util_ratio: float) -> str:
    if util_ratio >= 0.92:
        return 'critical'
    elif util_ratio >= 0.80:
        return 'stressed'
    else:
        return 'safe'


def risk_color(level: str) -> str:
    return {'safe': '#22c55e', 'stressed': '#f59e0b', 'critical': '#ef4444'}.get(level, '#94a3b8')


def risk_label(level: str) -> str:
    return {'safe': '🟢 Safe', 'stressed': '🟡 Stressed', 'critical': '🔴 Critical'}.get(level, '⚪ Unknown')
