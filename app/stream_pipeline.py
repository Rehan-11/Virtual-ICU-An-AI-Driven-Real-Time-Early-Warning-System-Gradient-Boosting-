# app/stream_pipeline.py
import pandas as pd
from datetime import timedelta
import numpy as np

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare patient vitals data from CSV"""
    try:
        df = pd.read_csv(csv_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['patient_id', 'timestamp'])
        return df.reset_index(drop=True)
    except FileNotFoundError:
        return pd.DataFrame(columns=['timestamp', 'patient_id', 'HR', 'SBP', 'DBP', 'MAP', 'SpO2', 'RR'])

def get_patient_ids(df: pd.DataFrame):
    """Get list of unique patient IDs"""
    if df.empty:
        return ['No patients']
    return sorted(df['patient_id'].unique().tolist())

def get_patient_df(df: pd.DataFrame, patient_id):
    """Filter data for specific patient"""
    if df.empty:
        return df
    return df[df['patient_id'] == patient_id].reset_index(drop=True)

def get_window(df_patient: pd.DataFrame, idx: int, window_seconds: int = 300):
    """Get rolling window of data up to current index"""
    if df_patient.empty or idx < 0:
        return df_patient.head(0)

    idx = min(idx, len(df_patient) - 1)

    if 'timestamp' in df_patient.columns and len(df_patient) > 0:
        current_time = df_patient.loc[idx, 'timestamp']
        start_time = current_time - pd.Timedelta(seconds=window_seconds)
        window_df = df_patient[
            (df_patient['timestamp'] <= current_time) & 
            (df_patient['timestamp'] >= start_time)
        ]
    else:
        start_idx = max(0, idx - window_seconds + 1)
        window_df = df_patient.iloc[start_idx:idx+1]

    return window_df

def calculate_trends(window: pd.DataFrame, vital_col: str) -> dict:
    """Calculate trend statistics for a vital sign"""
    if window.empty or vital_col not in window.columns:
        return {'slope': 0, 'variability': 0, 'latest': np.nan}

    values = window[vital_col].dropna()
    if len(values) < 2:
        return {'slope': 0, 'variability': 0, 'latest': values.iloc[-1] if len(values) > 0 else np.nan}

    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    slope = coeffs[0]

    variability = values.std() / values.mean() if values.mean() != 0 else 0

    return {
        'slope': slope,
        'variability': variability,
        'latest': values.iloc[-1]
    }
