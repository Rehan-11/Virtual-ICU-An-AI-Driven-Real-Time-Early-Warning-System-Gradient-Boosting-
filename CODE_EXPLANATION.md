# Code Explanation - Virtual ICU Monitor (Complete Line-by-Line Guide)

This document provides a detailed explanation of every line of code in the Virtual ICU Monitor project.

---

## Table of Contents
1. [run_icu.py](#run_icupy)
2. [app/stream_pipeline.py](#appstream_pipelinepy)
3. [app/ml_models.py](#appml_modelspy)
4. [app/model_training.py](#appmodel_trainingpy)
5. [app/model.py](#appmodelpy)
6. [app/ui_app.py](#appui_apppy)

---

## run_icu.py

### Purpose
Entry point script that launches the Streamlit web application for the Virtual ICU Monitor.

```python
# Line 1-2: File header comment
# run_icu.py
# Virtual ICU Monitor Launcher
```
Documents the file name and purpose.

```python
# Line 3: Import subprocess module for running system commands
import subprocess
```
`subprocess` allows Python to execute shell commands (like running Streamlit).

```python
# Line 4: Import sys module for system-specific parameters
import sys
```
`sys` provides access to Python interpreter variables and functions.

```python
# Line 5: Import os module for operating system operations (path handling, environment)
import os
```
`os` module allows interaction with the operating system (changing directories, file operations).

```python
# Line 7-8: Define the main function that launches the application
def main():
    """Launch the Virtual ICU Monitor"""
```
Function definition with docstring explaining what it does.

```python
# Line 9: Print startup message
print("Starting Virtual ICU Monitor...")
```
Informs user that the app is starting.

```python
# Line 10: Print status message
print("Loading patient data and initializing dashboard...")
```
Provides feedback about initialization progress.

```python
# Line 12: Change to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
```
- `os.path.abspath(__file__)` = get absolute path to current script
- `os.path.dirname()` = get directory containing the script
- `os.chdir()` = change working directory to that location
- Ensures relative paths work correctly for data loading.

```python
# Line 14-24: Try-except block for error handling
try:
```
Starts error handling block.

```python
# Line 15-21: Run Streamlit application
subprocess.run([
    sys.executable, "-m", "streamlit", "run", 
    "app/ui_app.py",
    "--server.headless", "false",
    "--server.runOnSave", "true",
    "--browser.gatherUsageStats", "false"
])
```
- `sys.executable` = path to Python interpreter
- `"-m", "streamlit"` = run Streamlit module
- `"run"` = run a Streamlit app
- `"app/ui_app.py"` = the app file to run
- `"--server.headless", "false"` = show browser UI
- `"--server.runOnSave", "true"` = auto-reload on file changes
- `"--browser.gatherUsageStats", "false"` = disable analytics

```python
# Line 22-23: Catch keyboard interrupt (Ctrl+C)
except KeyboardInterrupt:
    print("\nVirtual ICU Monitor stopped.")
```
Graceful shutdown message when user stops the app.

```python
# Line 25-26: Script entry point
if __name__ == "__main__":
    main()
```
Only runs `main()` if script is executed directly (not imported as module).

---

## app/stream_pipeline.py

### Purpose
Data loading and preprocessing pipeline for patient vital signs data.

```python
# Line 1-3: Imports
import pandas as pd
from datetime import timedelta
import numpy as np
```
- `pandas` = data manipulation library
- `timedelta` = for date/time calculations
- `numpy` = numerical operations

```python
# Line 5-11: Function to load CSV data
def load_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare patient vitals data from CSV"""
    try:
        df = pd.read_csv(csv_path)
```
- Function signature with type hints: takes string path, returns DataFrame
- `pd.read_csv()` reads CSV file into DataFrame

```python
# Line 9-10: Process timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
```
- Checks if 'timestamp' column exists
- Converts to datetime format for time operations

```python
# Line 11: Sort data by patient and timestamp
            df = df.sort_values(['patient_id', 'timestamp'])
```
Orders data chronologically for each patient.

```python
# Line 12: Return processed dataframe
        return df.reset_index(drop=True)
```
`reset_index(drop=True)` resets row numbers to 0,1,2...

```python
# Line 13-14: Handle file not found
    except FileNotFoundError:
        return pd.DataFrame(columns=['timestamp', 'patient_id', 'HR', 'SBP', 'DBP', 'MAP', 'SpO2', 'RR'])
```
Returns empty DataFrame with column names if file not found.

```python
# Line 16-19: Get unique patient IDs
def get_patient_ids(df: pd.DataFrame):
    """Get list of unique patient IDs"""
    if df.empty:
        return ['No patients']
    return sorted(df['patient_id'].unique().tolist())
```
- `df['patient_id'].unique()` = all unique patient IDs
- `tolist()` = convert to Python list
- `sorted()` = alphabetical order
- Returns empty list message if no data

```python
# Line 21-24: Filter data for specific patient
def get_patient_df(df: pd.DataFrame, patient_id):
    """Filter data for specific patient"""
    if df.empty:
        return df
    return df[df['patient_id'] == patient_id].reset_index(drop=True)
```
Returns all rows for one patient, resets row indices.

```python
# Line 26-40: Get rolling time window of data
def get_window(df_patient: pd.DataFrame, idx: int, window_seconds: int = 300):
    """Get rolling window of data up to current index"""
    if df_patient.empty or idx < 0:
        return df_patient.head(0)
```
- Takes patient dataframe, current index, and window size (default 5 min)
- Returns empty dataframe if input invalid

```python
# Line 32: Clamp index to valid range
    idx = min(idx, len(df_patient) - 1)
```
Ensures index doesn't exceed dataframe length.

```python
# Line 34-36: Get time-based window if timestamps exist
    if 'timestamp' in df_patient.columns and len(df_patient) > 0:
        current_time = df_patient.loc[idx, 'timestamp']
        start_time = current_time - pd.Timedelta(seconds=window_seconds)
```
- Gets the timestamp at current index
- Calculates start time (current - window size)

```python
# Line 37-40: Filter data within time window
        window_df = df_patient[
            (df_patient['timestamp'] <= current_time) & 
            (df_patient['timestamp'] >= start_time)
        ]
```
Boolean indexing: keeps rows between start_time and current_time.

```python
# Line 41-43: Alternative: use index-based window
    else:
        start_idx = max(0, idx - window_seconds + 1)
        window_df = df_patient.iloc[start_idx:idx+1]
```
- If no timestamps, use row positions instead
- `iloc[]` = index location (by position)

```python
# Line 45: Return window data
    return window_df
```

```python
# Line 47-53: Calculate trend statistics
def calculate_trends(window: pd.DataFrame, vital_col: str) -> dict:
    """Calculate trend statistics for a vital sign"""
    if window.empty or vital_col not in window.columns:
        return {'slope': 0, 'variability': 0, 'latest': np.nan}
```
Returns default values if invalid input.

```python
# Line 54-56: Get numeric values, remove NaN
    values = window[vital_col].dropna()
    if len(values) < 2:
        return {'slope': 0, 'variability': 0, 'latest': values.iloc[-1] if len(values) > 0 else np.nan}
```
Needs at least 2 points for trend calculation.

```python
# Line 57-60: Calculate slope (rate of change)
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    slope = coeffs[0]
```
- `np.arange()` = create array [0,1,2,...]
- `np.polyfit()` = fit polynomial (degree 1 = line)
- Returns [slope, intercept]; we take slope

```python
# Line 61-62: Calculate variability (coefficient of variation)
    variability = values.std() / values.mean() if values.mean() != 0 else 0
```
Standard deviation divided by mean = relative variability.

```python
# Line 64-68: Return statistics
    return {
        'slope': slope,
        'variability': variability,
        'latest': values.iloc[-1]
    }
```
Returns dictionary with trend metrics.

---

## app/ml_models.py

### Purpose
Defines Gradient Boosting ML model classes for risk prediction.

```python
# Line 1-7: Imports
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
```
- `typing` = type hints
- `xgboost` = gradient boosting library
- `StandardScaler` = normalize features (not used here but available)
- `warnings.filterwarnings('ignore')` = suppress warning messages

```python
# Line 9-13: Base ML model class
class MLRiskPredictor:
    """Base class for ML-based risk prediction using Gradient Boosting"""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_type = model_type
        self.is_trained = False
```
Initialize with empty model, flag for training status.

```python
# Line 15-56: Extract features from patient window
def extract_features(self, window: pd.DataFrame) -> np.ndarray:
    """Extract features from patient data window"""
    if window.empty:
        return np.zeros((1, len(self.feature_names)))
    
    latest = window.iloc[-1]
    features = []
    
    # Basic vitals (7 features)
    features.append(latest.get('HR', 80))
    features.append(latest.get('SBP', 120))
    features.append(latest.get('DBP', 80))
    features.append(latest.get('MAP', 90))
    features.append(latest.get('SpO2', 98))
    features.append(latest.get('RR', 16))
    features.append(latest.get('Temp', 37))
```
- `latest = window.iloc[-1]` = get most recent row
- `latest.get('HR', 80)` = get HR, default 80 if missing
- Add 7 basic vital signs to features list

```python
# Line 33-48: Calculate trend features (8 features)
    if len(window) >= 3:
        hr_values = window['HR'].fillna(80).values[-3:]
        features.append(np.mean(hr_values))
        features.append(np.std(hr_values) if len(hr_values) > 1 else 0)
```
- Get last 3 HR values (fill missing with 80)
- Append mean and standard deviation of HR
- Repeat for RR, SpO2, Temp (8 trend features total)

```python
# Line 49-56: Static and deviation features
    features.append(latest.get('age', 50))
    
    features.append(abs(latest.get('HR', 80) - 70))
    features.append(abs(latest.get('SBP', 120) - 120))
    features.append(abs(latest.get('SpO2', 98) - 97))
    features.append(abs(latest.get('RR', 16) - 14))
    
    return np.array(features).reshape(1, -1)
```
- Age feature
- 4 deviation features (absolute difference from normal ranges)
- Convert list to numpy array with shape (1, 20)

```python
# Line 58-68: Make prediction
def predict(self, window: pd.DataFrame) -> float:
    """Predict risk score (0.0-1.0)"""
    if not self.is_trained or self.model is None:
        return 0.0
    
    try:
        features = self.extract_features(window)
        prediction = self.model.predict(features)[0]
        return float(np.clip(prediction, 0.0, 1.0))
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0
```
- Check if model is trained
- Extract features
- Use model to predict
- `np.clip()` = ensure output between 0 and 1

```python
# Line 70-73: Load pre-trained model
def set_trained_model(self, model):
    """Set a pre-trained model"""
    self.model = model
    self.is_trained = True
```
Load existing trained model from disk.

```python
# Line 76-82: Sepsis-specific predictor
class SepsisPredictor(MLRiskPredictor):
    """Gradient Boosting model for sepsis risk prediction"""
    
    def __init__(self):
        super().__init__('xgboost')
        self.feature_names = [
            'HR', 'SBP', 'DBP', 'MAP', 'SpO2', 'RR', 'Temp',
            'HR_mean_3', 'HR_std_3', 'RR_mean_3', 'RR_std_3',
            'SpO2_mean_3', 'SpO2_std_3', 'Temp_mean_3', 'Temp_std_3',
            'age', 'HR_deviation', 'SBP_deviation', 'SpO2_deviation', 'RR_deviation'
        ]
```
- `super().__init__()` = call parent class constructor
- Define 20 feature names for documentation

```python
# Line 84-92: Sepsis prediction with clinical adjustment
def predict(self, window: pd.DataFrame) -> float:
    """Predict sepsis risk with ML model"""
    risk = super().predict(window)
    
    if not window.empty:
        latest = window.iloc[-1]
        temp = latest.get('Temp', 37)
        if (temp < 36.0 or temp >= 38.0) and risk < 0.3:
            risk = min(risk + 0.2, 1.0)
    
    return risk
```
- Get ML prediction using parent class
- If fever/hypothermia present but ML risk low, boost it by 0.2
- This adds clinical knowledge to ML predictions (hybrid approach)

```python
# Line 95-103: Cardiac predictor class
class CardiacPredictor(MLRiskPredictor):
    """Gradient Boosting model for cardiac arrest risk prediction"""
    # Same structure as SepsisPredictor
```
Similar structure for cardiac predictions.

```python
# Line 106-110: Cardiac prediction adjustment
def predict(self, window: pd.DataFrame) -> float:
    """Predict cardiac arrest risk with ML model"""
    risk = super().predict(window)
    
    if not window.empty:
        latest = window.iloc[-1]
        rr = latest.get('RR', 16)
        if rr >= 32 and risk < 0.4:
            risk = min(risk + 0.3, 1.0)
    
    return risk
```
If respiratory rate very high and risk low, boost by 0.3.

```python
# Line 114-118: Respiratory predictor class
class RespiratoryPredictor(MLRiskPredictor):
    """Gradient Boosting model for respiratory failure risk prediction"""
    # Same structure
```

```python
# Line 121-125: Respiratory prediction adjustment
def predict(self, window: pd.DataFrame) -> float:
    """Predict respiratory failure risk with ML model"""
    risk = super().predict(window)
    
    if not window.empty:
        latest = window.iloc[-1]
        spo2 = latest.get('SpO2', 98)
        if spo2 <= 90 and risk < 0.4:
            risk = min(risk + 0.35, 1.0)
    
    return risk
```
If low oxygen saturation and risk low, boost by 0.35.

---

## app/model_training.py

### Purpose
Trains Gradient Boosting models on patient vital signs data.

```python
# Line 1-11: Imports
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
```
Standard ML libraries for training and model persistence.

```python
# Line 13-47: Create synthetic labels from clinical rules
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
```
- Loop through each row in dataframe
- Extract vital signs
- If fever + tachycardia + tachypnea + hypotension = high sepsis risk (0.6)
- Otherwise if just fever + tachycardia + tachypnea = medium (0.4)

```python
# Line 22-28: Cardiac risk indicators
        # Cardiac risk indicators
        if rr >= 32 and sbp <= 100 and hr >= 110:
            risk = max(risk, 0.7)
        elif rr >= 28 and (sbp <= 90 or map_val <= 65):
            risk = max(risk, 0.5)
```
High RR + low BP + high HR = cardiac risk (0.7 or 0.5).

```python
# Line 30-36: Respiratory risk indicators
        # Respiratory risk indicators
        if spo2 <= 85:
            risk = max(risk, 0.8)
        elif spo2 <= 90 and rr >= 25:
            risk = max(risk, 0.6)
        elif spo2 <= 92 and rr >= 30:
            risk = max(risk, 0.5)
```
Low oxygen + high RR = respiratory risk (0.8, 0.6, or 0.5).

```python
# Line 38-41: Append and return labels
        labels.append(risk)
    
    return pd.Series(labels, index=df.index)
```
Return series of risk labels for all rows.

```python
# Line 44-67: Extract training features
def extract_training_features(df: pd.DataFrame, patient_id: str = None) -> Tuple[np.ndarray, pd.Series]:
    """Extract features and labels for training"""
    if patient_id:
        df = df[df['patient_id'] == patient_id].reset_index(drop=True)
```
Filter to one patient if specified.

```python
# Line 48-66: Feature extraction loop
    for idx in range(len(df)):
        start_idx = max(0, idx - 4)
        window = df.iloc[start_idx:idx+1]
        
        latest = window.iloc[-1]
        row_features = []
        
        # Basic vitals (7 features)
        row_features.append(latest.get('HR', 80))
        # ... (same as ml_models.py)
```
Extract features for each time point using last 5 records as window.

```python
# Line 68-73: Return features and empty labels
        features.append(row_features)
        labels.append(0)  # Placeholder, will be replaced with synthetic labels
    
    return np.array(features), pd.Series(np.array(labels))
```
Return stacked features array and placeholder labels.

```python
# Line 76-86: Train a single risk model
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
```
Loop through each patient, extract features.

```python
# Line 87-95: Stack features and create labels
    X = np.vstack(all_features)
    y_initial = np.concatenate(all_labels)
    
    # Create synthetic labels based on clinical rules
    y = create_synthetic_labels(df)
    y_array = y.values
    
    # Add variability to make it more realistic
    y_array = np.clip(y_array + np.random.normal(0, 0.05, len(y_array)), 0, 1)
```
- `np.vstack()` = stack feature arrays vertically
- Create synthetic labels
- Add small random noise to make more realistic

```python
# Line 97-100: Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_array, test_size=test_size, random_state=42
    )
```
80% train, 20% test; `random_state=42` for reproducibility.

```python
# Line 102-110: Create XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
```
XGBoost regressor with hyperparameters:
- 100 trees
- Max depth 5 (prevents overfitting)
- Learning rate 0.1
- Mean squared error objective
- No verbose output

```python
# Line 112-113: Train model
    model.fit(X_train, y_train, verbose=False)
```
Fit on training data.

```python
# Line 115-121: Evaluate and report
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{model_name} Model Trained")
    print(f"   Train R2 Score: {train_score:.4f}")
    print(f"   Test R2 Score: {test_score:.4f}")
    
    return model
```
Calculate R² scores (0-1, higher is better) and print metrics.

```python
# Line 124-156: Train and save all models
def train_and_save_all_models(csv_path: str, output_dir: str = "data/models"):
    """Train all three risk models and save them"""
    
    print("Loading training data...")
    df = pd.read_csv(csv_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['patient_id', 'timestamp'])
```
Load data and sort chronologically.

```python
# Line 132-138: Create models
    os.makedirs(output_dir, exist_ok=True)
    
    models = {
        'sepsis_model': train_risk_model(df, 'Sepsis Risk'),
        'cardiac_model': train_risk_model(df, 'Cardiac Arrest Risk'),
        'respiratory_model': train_risk_model(df, 'Respiratory Failure Risk')
    }
```
Create output directory and train all three models.

```python
# Line 140-145: Save models to disk
    for model_name, model in models.items():
        save_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(model, save_path)
        print(f"Saved {model_name} to {save_path}")
```
Save each trained model as pickle file.

```python
# Line 147-156: Load models from disk
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
```
Load pickle files into memory for predictions.

---

## app/model.py

### Purpose
Risk assessment logic combining ML models with clinical rules.

```python
# Line 1-8: Imports and module-level variables
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

_ml_models = None
```
`_ml_models` = global variable for caching loaded models (lazy loading).

```python
# Line 10-33: Load ML models
def _load_ml_models():
    """Load pre-trained ML models on first call"""
    global _ml_models
    if _ml_models is not None:
        return _ml_models
```
Check if models already loaded (avoid reloading).

```python
# Line 15-16: Build path to model directory
    _ml_models = {}
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
```
Navigate from model.py -> app -> project_root -> data/models.

```python
# Line 18-27: Load each model from disk
    model_files = {
        'sepsis': 'sepsis_model.pkl',
        'cardiac': 'cardiac_model.pkl',
        'respiratory': 'respiratory_model.pkl'
    }
    
    for model_key, model_file in model_files.items():
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            try:
                _ml_models[model_key] = joblib.load(model_path)
            except Exception as e:
                print(f"Warning: Could not load {model_key} model: {e}")
    
    return _ml_models
```
Try to load each model, skip if missing/error.

```python
# Line 35-71: Extract features for ML prediction
def _extract_ml_features(window: pd.DataFrame) -> np.ndarray:
    """Extract features for ML model prediction"""
    if window.empty:
        return np.zeros((1, 20))
    
    latest = window.iloc[-1]
    features = []
    
    # Basic vitals (7 features)
    features.append(latest.get('HR', 80))
    features.append(latest.get('SBP', 120))
    features.append(latest.get('DBP', 80))
    features.append(latest.get('MAP', 90))
    features.append(latest.get('SpO2', 98))
    features.append(latest.get('RR', 16))
    features.append(latest.get('Temp', 37))
```
Extract 7 basic vital signs (see ml_models.py for details).

```python
# Line 45-62: Trend features
    # Trends (if enough data points)
    if len(window) >= 3:
        hr_values = window['HR'].fillna(80).values[-3:]
        features.append(np.mean(hr_values))
        features.append(np.std(hr_values) if len(hr_values) > 1 else 0)
        # ... (repeat for RR, SpO2, Temp)
    else:
        features.extend([0] * 8)
```
Calculate trend features for HR, RR, SpO2, Temp.

```python
# Line 67-72: Age and deviations
    features.append(latest.get('age', 50))
    
    features.append(abs(latest.get('HR', 80) - 70))
    features.append(abs(latest.get('SBP', 120) - 120))
    features.append(abs(latest.get('SpO2', 98) - 97))
    features.append(abs(latest.get('RR', 16) - 14))
    
    return np.array(features).reshape(1, -1)
```
Add age and deviation features, return as (1, 20) array.

```python
# Line 74-86: Get ML prediction
def _get_ml_prediction(window: pd.DataFrame, model_key: str) -> float:
    """Get ML model prediction for risk score"""
    models = _load_ml_models()
    
    if model_key not in models:
        return 0.0
    
    try:
        features = _extract_ml_features(window)
        model = models[model_key]
        prediction = model.predict(features)[0]
        return float(np.clip(prediction, 0.0, 1.0))
    except Exception as e:
        print(f"ML prediction error: {e}")
        return 0.0
```
Load models, extract features, make prediction, clip to [0, 1].

```python
# Line 89-103: Clinical scoring system
class EarlyWarningScores:
    @staticmethod
    def calculate_news2(vitals: pd.Series) -> Tuple[int, Dict]:
        """Calculate NEWS2 score with component breakdown"""
        score = 0
        components = {}
        
        # Respiratory Rate scoring
        rr = vitals.get('RR', np.nan)
        if pd.notna(rr):
            if rr <= 8:
                rr_score = 3
            elif rr <= 11:
                rr_score = 1
            elif rr <= 20:
                rr_score = 0
            elif rr <= 24:
                rr_score = 2
            else:
                rr_score = 3
        else:
            rr_score = 0
        score += rr_score
        components['RR'] = rr_score
```
NEWS2 scoring system for clinical assessment:
- RR ≤8: severe (3 points)
- RR 9-11: abnormal (1 point)
- RR 12-20: normal (0 points)
- RR 21-24: abnormal (2 points)
- RR ≥25: severe (3 points)

Continues similarly for SpO2, BP, HR, Temperature components. Maximum score is 20.

```python
# Line 105-128: qSOFA scoring
    @staticmethod
    def calculate_qsofa(vitals: pd.Series) -> Tuple[int, Dict]:
        """Calculate qSOFA score with component breakdown"""
        score = 0
        components = {}
        
        # Respiratory Rate ≥22 (1 point)
        rr = vitals.get('RR', np.nan)
        rr_score = 1 if pd.notna(rr) and rr >= 22 else 0
        score += rr_score
        components['RR ≥22'] = rr_score
        
        # Systolic BP ≤100 (1 point)
        sbp = vitals.get('SBP', np.nan)
        sbp_score = 1 if pd.notna(sbp) and sbp <= 100 else 0
        score += sbp_score
        components['SBP ≤100'] = sbp_score
        
        # Altered mental status (assume normal for synthetic data)
        ams_score = 0
        components['Altered Mental Status'] = ams_score
        
        return score, components
```
qSOFA (Quick SOFA) for sepsis detection: RR, SBP, mental status. Max 3 points.

```python
# Line 130-176: Sepsis risk calculation
def sepsis_risk_score(window: pd.DataFrame) -> float:
    """Calculate sepsis-specific risk score using ML model"""
    if window.empty:
        return 0.0
    
    latest = window.iloc[-1]
    
    # Primary sepsis indicator: HIGH FEVER (>=39°C) + abnormal vitals
    if temp >= 39.0:
        fever_score = 0.4
        hr_score = 0.2 if hr >= 100 else 0.1
        rr_score = 0.2 if rr >= 22 else 0.1
        hypotension_score = 0.0
        if sbp <= 90 or map_val <= 65:
            hypotension_score = 0.3
        elif sbp <= 100:
            hypotension_score = 0.2
        
        base_score = fever_score + hr_score + rr_score + hypotension_score
        return min(base_score, 1.0)
```
- High fever (≥39°C): +0.4
- High HR (≥100): +0.2, else +0.1
- High RR (≥22): +0.2, else +0.1
- Low BP: +0.3 or +0.2
- Max score 1.0

```python
# Line 162-169: Moderate fever handling
    if temp >= 38.0:
        fever_score = 0.25
        hr_score = 0.2 if hr >= 100 else 0.0
        rr_score = 0.2 if rr >= 22 else 0.0
        hypotension_score = 0.1 if sbp <= 100 else 0.0
        
        base_score = fever_score + hr_score + rr_score + hypotension_score
        return min(base_score, 1.0)
```
Lower thresholds for moderate fever (38-39°C).

```python
# Line 171-176: Hypothermia handling
    if temp <= 36.0 and hr >= 90 and rr >= 20:
        return 0.5
    
    return 0.0
```
Hypothermia + tachycardia + tachypnea = 0.5 risk.

```python
# Line 178-222: Cardiac arrest risk calculation
def cardiac_risk_score(window: pd.DataFrame) -> float:
    """Calculate cardiac arrest risk score using ML model"""
    if window.empty:
        return 0.0
    
    latest = window.iloc[-1]
    
    # Tachycardia (strong indicator of cardiac stress)
    if hr >= 120:
        score += 0.4
    elif hr >= 110:
        score += 0.3
    elif hr <= 60:
        score += 0.2
```
- HR ≥120: +0.4
- HR 110-119: +0.3
- HR ≤60 (bradycardia): +0.2

```python
# Line 193-200: Respiratory rate component
    # Elevated respiratory rate (indicator of cardiac compensation)
    if rr >= 28:
        score += 0.4
    elif rr >= 24:
        score += 0.35
    elif rr >= 20:
        score += 0.3
```
High RR indicates cardiac compromise.

```python
# Line 202-207: Blood pressure component
    # Blood pressure instability is critical
    if sbp <= 100 or map_val <= 65:
        score += 0.35
    elif sbp <= 110 or map_val <= 75:
        score += 0.3
```
Low BP critical for cardiac arrest risk.

```python
# Line 209-222: Final scoring
    if dbp <= 50:
        score += 0.15
    
    if age >= 75:
        score += 0.1
    elif age >= 65:
        score += 0.05
    
    if temp >= 39.0:
        score *= 0.6
    
    return min(score, 1.0)
```
- Low DBP: +0.15
- Elderly: +0.1 or +0.05
- High fever suppresses cardiac risk (likely sepsis instead)

```python
# Line 224-276: Respiratory failure risk calculation
def respiratory_risk_score(window: pd.DataFrame) -> float:
    """Calculate respiratory failure risk score using ML model"""
    if window.empty:
        return 0.0
    
    latest = window.iloc[-1]
    
    # Hypoxemia (primary respiratory failure indicator)
    if spo2 <= 80:
        score += 0.5
    elif spo2 <= 85:
        score += 0.4
    elif spo2 <= 90:
        score += 0.3
    elif spo2 <= 92:
        score += 0.2
    elif spo2 <= 94:
        score += 0.1
```
SpO2 ≤94% = concerning, ≤80% = severe.

```python
# Line 252-276: Tachypnea and trends
    if rr >= 40:
        score += 0.3
    elif rr >= 35:
        score += 0.25
    elif rr >= 30:
        score += 0.2
    elif rr >= 25:
        score += 0.1
    
    if hr >= 120 and spo2 <= 92:
        score += 0.2
    elif hr >= 100 and spo2 <= 90:
        score += 0.1
    
    # Trending SpO2 (if declining)
    if len(window) >= 5 and 'SpO2' in window.columns:
        spo2_values = window['SpO2'].fillna(98).values
        if len(spo2_values) >= 3:
            recent_trend = np.mean(spo2_values[-3:]) - np.mean(spo2_values[-6:-3]) if len(spo2_values) >= 6 else 0
            if recent_trend < -2:
                score += 0.2
            elif recent_trend < -1:
                score += 0.1
    
    if temp >= 38.5:
        score *= 0.8
    
    return min(score, 1.0)
```
- High RR: +0.3 to +0.1
- Tachycardia + low SpO2: +0.2 or +0.1
- Declining SpO2 trend: +0.2 or +0.1
- High fever suppresses respiratory risk

```python
# Line 278-406: Comprehensive assessment
def get_comprehensive_assessment(window: pd.DataFrame) -> Dict:
    """Get comprehensive patient risk assessment with proper class separation"""
    if window.empty:
        return {
            'overall_risk': 0.0,
            'primary_concern': 'Stable',
            'sepsis': {'risk': 0.0, 'prediction': 'Low Risk', ...},
            # ... all empty/default values
        }
```
Return structure if no data.

```python
# Line 287-302: Calculate all scores
    latest_vitals = window.iloc[-1]
    
    news2_score, news2_components = EarlyWarningScores.calculate_news2(latest_vitals)
    qsofa_score, qsofa_components = EarlyWarningScores.calculate_qsofa(latest_vitals)
    
    sepsis_risk = sepsis_risk_score(window)
    cardiac_risk = cardiac_risk_score(window)
    respiratory_risk = respiratory_risk_score(window)
```
Calculate all risk scores.

```python
# Line 304-330: Determine primary concern
    risks = {
        'sepsis': sepsis_risk,
        'cardiac': cardiac_risk,
        'respiratory': respiratory_risk
    }
    
    primary_concern_key = max(risks, key=risks.get)
    overall_risk = max(risks.values())
    
    if overall_risk < 0.3:
        primary_concern = 'Stable'
    else:
        primary_concern = concern_mapping[primary_concern_key]
```
Find highest risk category, use as primary concern.

```python
# Line 332-363: Build assessment dictionary
    def get_prediction(risk_score):
        if risk_score >= 0.7:
            return 'High Risk'
        elif risk_score >= 0.4:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    assessment = {
        'overall_risk': overall_risk,
        'primary_concern': primary_concern,
        'sepsis': {
            'risk': sepsis_risk,
            'prediction': get_prediction(sepsis_risk),
            'confidence': min(sepsis_risk + 0.1, 1.0),
            'news2_score': news2_score,
            'qsofa_score': qsofa_score
        },
        # ... similar for cardiac and respiratory
    }
```
Convert risk scores to prediction categories.

```python
# Line 365-405: Generate recommendations
    if overall_risk >= 0.7:
        if primary_concern_key == 'sepsis':
            recommendations.extend([
                "IMMEDIATE: Initiate sepsis bundle protocol",
                "Obtain blood cultures before antibiotics",
                "Consider broad-spectrum antibiotics",
                "Check lactate and procalcitonin levels"
            ])
        # ... similar for cardiac and respiratory
    elif overall_risk >= 0.4:
        recommendations.extend([...])
    else:
        recommendations.append("Continue routine monitoring")
```
Generate clinical recommendations based on risk level and primary concern.

---

## app/ui_app.py

### Purpose
Streamlit web interface for the Virtual ICU Monitor dashboard.

```python
# Line 1-12: Imports
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.stream_pipeline import load_data, get_patient_ids, get_patient_df, get_window
from app.model import get_comprehensive_assessment, get_hard_alerts, get_risk_level_info, EarlyWarningScores
```
- `streamlit` = web framework
- `plotly` = interactive charts
- Add parent directory to path to import app modules

```python
# Line 18-24: Streamlit page configuration
st.set_page_config(
    page_title="Virtual ICU Monitor - AI Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
```
Set page title, icon, layout width, and sidebar state.

```python
# Line 26-115: Custom CSS styling
st.markdown("""
<style>
.disease-risk-high {
    background-color: #d32f2f;
    color: white;
    border: 3px solid #b71c1c;
    ...
}
# ... many more CSS classes
</style>
""", unsafe_allow_html=True)
```
Define custom HTML/CSS styling for risk levels and components.

```python
# Line 117-130: Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = 'P001'
# ... more session state variables
```
Streamlit session_state persists variables across page refreshes. Initializes variables first time.

```python
# Line 132-151: Main header
st.markdown("# 🏥 Virtual ICU Monitor - AI Disease Prediction System")
```
Displays main title.

```python
# Line 153-156: Sidebar - Patient selection
with st.sidebar:
    st.markdown("### Patient Selection")
    selected_patient = st.selectbox(
        "Select Patient",
        patient_ids,
        index=patient_ids.index(st.session_state.current_patient) if st.session_state.current_patient in patient_ids else 0
    )
```
Dropdown menu for patient selection, defaults to current selection.

```python
# Line 158-172: Sidebar - Simulation controls
    st.markdown("### Simulation Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Play", use_container_width=True):
            st.session_state.is_playing = True
```
Buttons for Play/Pause/Reset controls.

```python
# Line 174-183: Sidebar - Speed and window settings
    speed = st.slider(
        "Simulation Speed (min per refresh)",
        min_value=1, max_value=15,
        value=st.session_state.speed,
        step=1
    )
```
Sliders for simulation speed and analysis window size.

```python
# Line 185-195: Sidebar - Alert thresholds
    st.markdown("### Alert Thresholds")
    sepsis_threshold = st.slider(
        "Sepsis Alert Threshold",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05
    )
```
Sliders to adjust when alerts trigger.

```python
# Line 197-210: Load and process data
with st.spinner("Loading patient data..."):
    full_df = load_data("data/patient_vitals_enhanced.csv")
    patient_ids = get_patient_ids(full_df)
    patient_df = get_patient_df(full_df, selected_patient)
    st.session_state.data_loaded = True
```
Load CSV data, get patient IDs, filter for selected patient.

```python
# Line 212-230: Main layout - Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Real-Time Dashboard",
    "Risk Assessment",
    "Clinical Scores",
    "Recommendations",
    "About"
])
```
Create 5 tabs for different views.

```python
# Line 232-270: Tab 1 - Real-time dashboard
with tab1:
    st.markdown("## Real-Time Patient Monitoring")
    
    # Header info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Time", current_time.strftime("%H:%M:%S"))
    with col2:
        st.metric("Time in Simulation", f"{elapsed_minutes:.1f} min")
    with col3:
        st.metric("Status", simulation_status)
```
Display current time, elapsed time, simulation status in 3 columns.

```python
# Line 272-330: Vital signs display (3x2 grid)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        st.metric("Heart Rate", f"{latest_vitals['HR']:.1f}", f"{hr_change:+.1f}", delta_color="off")
    # ... similar for SBP, DBP, SpO2, RR, Temp
```
Display 6 vital signs with current value and change from previous.

```python
# Line 332-420: Risk indicator cards
    st.markdown("### Disease Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if assessment['sepsis']['risk'] >= 0.7:
            st.markdown(f"<div class='disease-risk-high'><h4>Sepsis Risk</h4><h2>{assessment['sepsis']['risk']:.1%}</h2>...</div>", unsafe_allow_html=True)
        elif assessment['sepsis']['risk'] >= 0.4:
            st.markdown(f"<div class='disease-risk-medium'><h4>Sepsis Risk</h4><h2>{assessment['sepsis']['risk']:.1%}</h2>...</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='disease-risk-low'><h4>Sepsis Risk</h4><h2>{assessment['sepsis']['risk']:.1%}</h2>...</div>", unsafe_allow_html=True)
```
Display risk cards for Sepsis, Cardiac, Respiratory with color coding based on risk level.

```python
# Line 422-450: Vital signs chart
    st.markdown("### Vital Signs Trends")
    
    # Create subplots for all vitals
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Heart Rate", "Systolic BP", "Diastolic BP", "SpO2", "Respiratory Rate", "Temperature"),
        specs=[[{"secondary_y": False}]*3] * 2
    )
```
Create 2x3 subplot layout for vital signs trends.

```python
# Line 452-475: Add traces to chart
    fig.add_trace(
        go.Scatter(x=window['timestamp'], y=window['HR'], name='HR', mode='lines', line=dict(color='red')),
        row=1, col=1
    )
    # ... similar for other vitals
```
Add line traces for each vital sign to corresponding subplot.

```python
# Line 477-485: Update layout
    fig.update_xaxes(title_text="Time", row=2, col=3)
    fig.update_yaxes(title_text="HR (bpm)", row=1, col=1)
    # ... similar for other axes
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
```
Set axis labels and chart height, display.

```python
# Line 487-510: Tab 2 - Risk assessment details
with tab2:
    st.markdown("## Risk Assessment Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Sepsis Risk")
        st.write(f"**Risk Score:** {assessment['sepsis']['risk']:.3f}")
        st.write(f"**Prediction:** {assessment['sepsis']['prediction']}")
        st.write(f"**NEWS2 Score:** {assessment['sepsis']['news2_score']}/20")
        st.write(f"**qSOFA Score:** {assessment['sepsis']['qsofa_score']}/3")
```
Display detailed risk scores and clinical scoring components.

```python
# Line 512-550: Risk gauge charts
    fig_risk = go.Figure(data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=assessment['sepsis']['risk'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sepsis Risk"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#90EE90"},
                    {'range': [30, 70], 'color': "#FFD700"},
                    {'range': [70, 100], 'color': "#FF6B6B"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        )
    ])
    st.plotly_chart(fig_risk, use_container_width=True)
```
Create gauge chart showing risk level with color zones.

```python
# Line 552-580: Tab 3 - Clinical scores
with tab3:
    st.markdown("## Clinical Scoring Systems")
    
    st.markdown("### NEWS2 (National Early Warning Score 2)")
    st.write(f"**Total Score:** {assessment['sepsis']['news2_score']}/20")
    st.write("**Interpretation:**")
    st.write("- 0-4: Low risk")
    st.write("- 5-6: Medium risk")
    st.write("- 7+: High risk")
    
    # Display component breakdown
    news2_components = {
        'Respiratory Rate': assessment['sepsis'].get('news2_rr', 0),
        'SpO2': assessment['sepsis'].get('news2_spo2', 0),
        'Blood Pressure': assessment['sepsis'].get('news2_bp', 0),
        'Heart Rate': assessment['sepsis'].get('news2_hr', 0),
        'Temperature': assessment['sepsis'].get('news2_temp', 0)
    }
```
Display NEWS2 scoring system with interpretation guide.

```python
# Line 582-610: Tab 4 - Recommendations
with tab4:
    st.markdown("## Clinical Recommendations")
    
    st.markdown("### Primary Concern")
    st.info(f"**{assessment['primary_concern']}**")
    
    st.markdown("### Recommended Actions")
    for i, rec in enumerate(assessment['recommendations'], 1):
        st.markdown(f"{i}. {rec}")
```
Display clinical recommendations based on risk assessment.

```python
# Line 612-650: Hard alerts
    if hard_alerts:
        st.markdown("### CRITICAL ALERTS")
        for alert in hard_alerts:
            st.error(f"🚨 {alert}")
```
Display critical hard alerts that require immediate attention.

```python
# Line 652-700: Tab 5 - About
with tab5:
    st.markdown("## About Virtual ICU Monitor")
    
    st.markdown("""
    ### Overview
    The Virtual ICU Monitor is an AI-powered system for real-time patient disease prediction...
    
    ### Machine Learning Approach
    - **Algorithm:** XGBoost (Gradient Boosting)
    - **Models:** Sepsis, Cardiac Arrest, Respiratory Failure
    - **Accuracy:** 95%+ on test data
    """)
```
Display information about the project and ML approach.

---

## Summary of Code Structure

### Data Flow
1. **run_icu.py** → Launches Streamlit app
2. **stream_pipeline.py** → Loads CSV, prepares data
3. **ml_models.py & model_training.py** → ML models (trained once, loaded in memory)
4. **model.py** → Calculates risk scores (combines ML + clinical rules)
5. **ui_app.py** → Displays results in web dashboard

### Key Algorithms
- **NEWS2 Scoring:** Clinical early warning system
- **qSOFA Scoring:** Sepsis risk assessment
- **Gradient Boosting:** ML predictions from vital signs
- **Feature Engineering:** 20 features extracted from patient vitals
- **Risk Stratification:** Low (<0.4), Medium (0.4-0.7), High (>0.7)

### File Organization
```
project/
├── run_icu.py                 # Entry point
├── requirements.txt           # Dependencies
├── data/
│   ├── patient_vitals_enhanced.csv    # Training data
│   └── models/                        # Trained ML models
│       ├── sepsis_model.pkl
│       ├── cardiac_model.pkl
│       └── respiratory_model.pkl
└── app/
    ├── stream_pipeline.py     # Data loading
    ├── ml_models.py           # ML model classes
    ├── model_training.py      # Training pipeline
    ├── model.py               # Risk calculation
    └── ui_app.py              # Streamlit UI
```

This completes the line-by-line explanation of all code in the Virtual ICU Monitor project.
