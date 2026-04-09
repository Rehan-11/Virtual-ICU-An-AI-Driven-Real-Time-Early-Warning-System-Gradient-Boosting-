# app/model_training.py - Training pipeline for Gradient Boosting models
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from typing import Tuple
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

def create_synthetic_labels(df: pd.DataFrame) -> pd.Series:
    """Create synthetic risk labels based on clinical indicators for training"""
    labels = []
    
    for idx, row in df.iterrows():
        risk = 0.0
        
        # Sepsis risk indicators
        temp = row.get('Temp', 37)
        hr = row.get('HR', 80)
        rr = row.get('RR', 16)
        sbp = row.get('SBP', 120)
        map_val = row.get('MAP', 80)
        
        if (temp >= 38.0 or temp <= 36.0) and hr >= 90 and rr >= 20:
            risk = max(risk, 0.6 if sbp <= 100 else 0.4)
        
        # Cardiac risk indicators
        if rr >= 32 and sbp <= 100 and hr >= 110:
            risk = max(risk, 0.7)
        elif rr >= 28 and (sbp <= 90 or map_val <= 65):
            risk = max(risk, 0.5)
        
        # Respiratory risk indicators
        spo2 = row.get('SpO2', 98)
        if spo2 <= 85:
            risk = max(risk, 0.8)
        elif spo2 <= 90 and rr >= 25:
            risk = max(risk, 0.6)
        elif spo2 <= 92 and rr >= 30:
            risk = max(risk, 0.5)
        
        labels.append(risk)
    
    return pd.Series(labels, index=df.index)


def extract_training_features(df: pd.DataFrame, patient_id: str = None) -> Tuple[np.ndarray, pd.Series]:
    """Extract features and labels for training"""
    if patient_id:
        df = df[df['patient_id'] == patient_id].reset_index(drop=True)
    
    features = []
    labels = []
    
    for idx in range(len(df)):
        # Create window of last 5 records
        start_idx = max(0, idx - 4)
        window = df.iloc[start_idx:idx+1]
        
        latest = window.iloc[-1]
        row_features = []
        
        # Basic vitals
        row_features.append(latest.get('HR', 80))
        row_features.append(latest.get('SBP', 120))
        row_features.append(latest.get('DBP', 80))
        row_features.append(latest.get('MAP', 90))
        row_features.append(latest.get('SpO2', 98))
        row_features.append(latest.get('RR', 16))
        row_features.append(latest.get('Temp', 37))
        
        # Trends
        if len(window) >= 3:
            hr_values = window['HR'].fillna(80).values[-3:]
            row_features.append(np.mean(hr_values))
            row_features.append(np.std(hr_values) if len(hr_values) > 1 else 0)
            
            rr_values = window['RR'].fillna(16).values[-3:]
            row_features.append(np.mean(rr_values))
            row_features.append(np.std(rr_values) if len(rr_values) > 1 else 0)
            
            spo2_values = window['SpO2'].fillna(98).values[-3:]
            row_features.append(np.mean(spo2_values))
            row_features.append(np.std(spo2_values) if len(spo2_values) > 1 else 0)
            
            temp_values = window['Temp'].fillna(37).values[-3:]
            row_features.append(np.mean(temp_values))
            row_features.append(np.std(temp_values) if len(temp_values) > 1 else 0)
        else:
            row_features.extend([0] * 8)
        
        # Age
        row_features.append(latest.get('age', 50))
        
        # Deviations
        row_features.append(abs(latest.get('HR', 80) - 70))
        row_features.append(abs(latest.get('SBP', 120) - 120))
        row_features.append(abs(latest.get('SpO2', 98) - 97))
        row_features.append(abs(latest.get('RR', 16) - 14))
        
        features.append(row_features)
        labels.append(0)  # Placeholder, will be replaced with synthetic labels
    
    return np.array(features), pd.Series(np.array(labels))


def train_risk_model(df: pd.DataFrame, model_name: str, test_size: float = 0.2) -> xgb.XGBRegressor:
    """Train a Gradient Boosting model for risk prediction"""
    
    print(f"\nTraining {model_name} model...")
    
    # Extract features from all patients
    all_features = []
    all_labels = []
    
    patient_ids = df['patient_id'].unique()
    for patient_id in patient_ids:
        features, _ = extract_training_features(df, patient_id)
        all_features.append(features)
        all_labels.append(_)
    
    X = np.vstack(all_features)
    y_initial = np.concatenate(all_labels)
    
    # Create synthetic labels based on clinical rules
    y = create_synthetic_labels(df)
    y_array = y.values
    
    # Add variability to make it more realistic
    y_array = np.clip(y_array + np.random.normal(0, 0.05, len(y_array)), 0, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_array, test_size=test_size, random_state=42
    )
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{model_name} Model Trained")
    print(f"   Train R2 Score: {train_score:.4f}")
    print(f"   Test R2 Score: {test_score:.4f}")
    
    return model


def train_and_save_all_models(csv_path: str, output_dir: str = "data/models"):
    """Train all three risk models and save them"""
    
    print("Loading training data...")
    df = pd.read_csv(csv_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['patient_id', 'timestamp'])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train models
    models = {
        'sepsis_model': train_risk_model(df, 'Sepsis Risk'),
        'cardiac_model': train_risk_model(df, 'Cardiac Arrest Risk'),
        'respiratory_model': train_risk_model(df, 'Respiratory Failure Risk')
    }
    
    # Save models
    for model_name, model in models.items():
        save_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(model, save_path)
        print(f"Saved {model_name} to {save_path}")
    
    print("\nAll models trained and saved!")
    return models


def load_models(model_dir: str = "data/models") -> dict:
    """Load pre-trained models from disk"""
    models = {}
    
    model_files = ['sepsis_model.pkl', 'cardiac_model.pkl', 'respiratory_model.pkl']
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            try:
                models[model_file.replace('.pkl', '')] = joblib.load(model_path)
                print(f"Loaded {model_file}")
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    
    return models


if __name__ == "__main__":
    # Train models
    train_and_save_all_models("data/patient_vitals_enhanced.csv")
