# app/ml_models.py - Gradient Boosting Models for Risk Prediction
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MLRiskPredictor:
    """Base class for ML-based risk prediction using Gradient Boosting"""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_type = model_type
        self.is_trained = False
    
    def extract_features(self, window: pd.DataFrame) -> np.ndarray:
        """Extract features from patient data window"""
        if window.empty:
            return np.zeros((1, len(self.feature_names)))
        
        latest = window.iloc[-1]
        features = []
        
        # Basic vitals
        features.append(latest.get('HR', 80))
        features.append(latest.get('SBP', 120))
        features.append(latest.get('DBP', 80))
        features.append(latest.get('MAP', 90))
        features.append(latest.get('SpO2', 98))
        features.append(latest.get('RR', 16))
        features.append(latest.get('Temp', 37))
        
        # Trends (if enough data points)
        if len(window) >= 3:
            hr_values = window['HR'].fillna(80).values[-3:]
            features.append(np.mean(hr_values))
            features.append(np.std(hr_values) if len(hr_values) > 1 else 0)
            
            rr_values = window['RR'].fillna(16).values[-3:]
            features.append(np.mean(rr_values))
            features.append(np.std(rr_values) if len(rr_values) > 1 else 0)
            
            spo2_values = window['SpO2'].fillna(98).values[-3:]
            features.append(np.mean(spo2_values))
            features.append(np.std(spo2_values) if len(spo2_values) > 1 else 0)
            
            temp_values = window['Temp'].fillna(37).values[-3:]
            features.append(np.mean(temp_values))
            features.append(np.std(temp_values) if len(temp_values) > 1 else 0)
        else:
            features.extend([0] * 8)  # 8 trend features if not enough data
        
        # Age
        features.append(latest.get('age', 50))
        
        # Deviation from normal ranges
        features.append(abs(latest.get('HR', 80) - 70))  # Normal HR ~70
        features.append(abs(latest.get('SBP', 120) - 120))  # Normal SBP ~120
        features.append(abs(latest.get('SpO2', 98) - 97))  # Normal SpO2 ~97
        features.append(abs(latest.get('RR', 16) - 14))  # Normal RR ~14
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, window: pd.DataFrame) -> float:
        """Predict risk score (0.0-1.0)"""
        if not self.is_trained or self.model is None:
            return 0.0
        
        try:
            features = self.extract_features(window)
            prediction = self.model.predict(features)[0]
            # Ensure output is between 0 and 1
            return float(np.clip(prediction, 0.0, 1.0))
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0
    
    def set_trained_model(self, model):
        """Set a pre-trained model"""
        self.model = model
        self.is_trained = True


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
    
    def predict(self, window: pd.DataFrame) -> float:
        """Predict sepsis risk with ML model"""
        risk = super().predict(window)
        
        # Add clinical rule adjustments for hybrid approach
        if not window.empty:
            latest = window.iloc[-1]
            temp = latest.get('Temp', 37)
            # Sepsis usually involves temperature abnormality
            if (temp < 36.0 or temp >= 38.0) and risk < 0.3:
                risk = min(risk + 0.2, 1.0)
        
        return risk


class CardiacPredictor(MLRiskPredictor):
    """Gradient Boosting model for cardiac arrest risk prediction"""
    
    def __init__(self):
        super().__init__('xgboost')
        self.feature_names = [
            'HR', 'SBP', 'DBP', 'MAP', 'SpO2', 'RR', 'Temp',
            'HR_mean_3', 'HR_std_3', 'RR_mean_3', 'RR_std_3',
            'SpO2_mean_3', 'SpO2_std_3', 'Temp_mean_3', 'Temp_std_3',
            'age', 'HR_deviation', 'SBP_deviation', 'SpO2_deviation', 'RR_deviation'
        ]
    
    def predict(self, window: pd.DataFrame) -> float:
        """Predict cardiac arrest risk with ML model"""
        risk = super().predict(window)
        
        # Add clinical rule adjustments
        if not window.empty:
            latest = window.iloc[-1]
            rr = latest.get('RR', 16)
            # High RR is strong cardiac arrest predictor
            if rr >= 32 and risk < 0.4:
                risk = min(risk + 0.3, 1.0)
        
        return risk


class RespiratoryPredictor(MLRiskPredictor):
    """Gradient Boosting model for respiratory failure risk prediction"""
    
    def __init__(self):
        super().__init__('xgboost')
        self.feature_names = [
            'HR', 'SBP', 'DBP', 'MAP', 'SpO2', 'RR', 'Temp',
            'HR_mean_3', 'HR_std_3', 'RR_mean_3', 'RR_std_3',
            'SpO2_mean_3', 'SpO2_std_3', 'Temp_mean_3', 'Temp_std_3',
            'age', 'HR_deviation', 'SBP_deviation', 'SpO2_deviation', 'RR_deviation'
        ]
    
    def predict(self, window: pd.DataFrame) -> float:
        """Predict respiratory failure risk with ML model"""
        risk = super().predict(window)
        
        # Add clinical rule adjustments
        if not window.empty:
            latest = window.iloc[-1]
            spo2 = latest.get('SpO2', 98)
            # Low SpO2 is primary respiratory failure indicator
            if spo2 <= 90 and risk < 0.4:
                risk = min(risk + 0.35, 1.0)
        
        return risk
