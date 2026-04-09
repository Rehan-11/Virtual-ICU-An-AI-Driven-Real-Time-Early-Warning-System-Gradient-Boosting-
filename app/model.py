# app/model.py - ML-Enhanced Risk Assessment with Gradient Boosting
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# Global ML model cache
_ml_models = None

def _load_ml_models():
    """Load pre-trained ML models on first call"""
    global _ml_models
    if _ml_models is not None:
        return _ml_models
    
    _ml_models = {}
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    
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


def _extract_ml_features(window: pd.DataFrame) -> np.ndarray:
    """Extract features for ML model prediction"""
    if window.empty:
        return np.zeros((1, 20))
    
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
        features.extend([0] * 8)
    
    # Age
    features.append(latest.get('age', 50))
    
    # Deviations
    features.append(abs(latest.get('HR', 80) - 70))
    features.append(abs(latest.get('SBP', 120) - 120))
    features.append(abs(latest.get('SpO2', 98) - 97))
    features.append(abs(latest.get('RR', 16) - 14))
    
    return np.array(features).reshape(1, -1)


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


class EarlyWarningScores:
    @staticmethod
    def calculate_news2(vitals: pd.Series) -> Tuple[int, Dict]:
        """Calculate NEWS2 score with component breakdown"""
        score = 0
        components = {}
        
        # Respiratory Rate
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
        
        # SpO2
        spo2 = vitals.get('SpO2', np.nan)
        if pd.notna(spo2):
            if spo2 <= 91:
                spo2_score = 3
            elif spo2 <= 93:
                spo2_score = 2
            elif spo2 <= 95:
                spo2_score = 1
            else:
                spo2_score = 0
        else:
            spo2_score = 0
        score += spo2_score
        components['SpO2'] = spo2_score
        
        # Blood Pressure (Systolic)
        sbp = vitals.get('SBP', np.nan)
        if pd.notna(sbp):
            if sbp <= 90:
                bp_score = 3
            elif sbp <= 100:
                bp_score = 2
            elif sbp <= 110:
                bp_score = 1
            elif sbp <= 219:
                bp_score = 0
            else:
                bp_score = 3
        else:
            bp_score = 0
        score += bp_score
        components['SBP'] = bp_score
        
        # Heart Rate
        hr = vitals.get('HR', np.nan)
        if pd.notna(hr):
            if hr <= 40:
                hr_score = 3
            elif hr <= 50:
                hr_score = 1
            elif hr <= 90:
                hr_score = 0
            elif hr <= 110:
                hr_score = 1
            elif hr <= 130:
                hr_score = 2
            else:
                hr_score = 3
        else:
            hr_score = 0
        score += hr_score
        components['HR'] = hr_score
        
        # Temperature
        temp = vitals.get('Temp', np.nan)
        if pd.notna(temp):
            if temp <= 35.0:
                temp_score = 3
            elif temp <= 36.0:
                temp_score = 1
            elif temp <= 38.0:
                temp_score = 0
            elif temp <= 39.0:
                temp_score = 1
            else:
                temp_score = 2
        else:
            temp_score = 0
        score += temp_score
        components['Temp'] = temp_score
        
        return score, components
    
    @staticmethod
    def calculate_qsofa(vitals: pd.Series) -> Tuple[int, Dict]:
        """Calculate qSOFA score with component breakdown"""
        score = 0
        components = {}
        
        # Respiratory Rate ≥22
        rr = vitals.get('RR', np.nan)
        rr_score = 1 if pd.notna(rr) and rr >= 22 else 0
        score += rr_score
        components['RR ≥22'] = rr_score
        
        # Systolic BP ≤100
        sbp = vitals.get('SBP', np.nan)
        sbp_score = 1 if pd.notna(sbp) and sbp <= 100 else 0
        score += sbp_score
        components['SBP ≤100'] = sbp_score
        
        # Altered mental status (assume normal for synthetic data)
        ams_score = 0
        components['Altered Mental Status'] = ams_score
        
        return score, components

def sepsis_risk_score(window: pd.DataFrame) -> float:
    """Calculate sepsis-specific risk score using ML model"""
    if window.empty:
        return 0.0
    
    latest = window.iloc[-1]
    
    # Sepsis requires fever/hypothermia + tachycardia/tachypnea + hypotension
    temp = latest.get('Temp', 37.0)
    hr = latest.get('HR', 80)
    rr = latest.get('RR', 16)
    sbp = latest.get('SBP', 120)
    map_val = latest.get('MAP', 80)
    
    # Primary sepsis indicator: HIGH FEVER (>=39°C) + abnormal vitals
    if temp >= 39.0:
        # With high fever, check for SIRS criteria
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
    
    # Secondary: moderate fever (38-39°C) with strong indicators
    if temp >= 38.0:
        fever_score = 0.25
        hr_score = 0.2 if hr >= 100 else 0.0
        rr_score = 0.2 if rr >= 22 else 0.0
        hypotension_score = 0.1 if sbp <= 100 else 0.0
        
        base_score = fever_score + hr_score + rr_score + hypotension_score
        return min(base_score, 1.0)
    
    # Hypothermia scenario
    if temp <= 36.0 and hr >= 90 and rr >= 20:
        return 0.5
    
    return 0.0

def cardiac_risk_score(window: pd.DataFrame) -> float:
    """Calculate cardiac arrest risk score using ML model"""
    if window.empty:
        return 0.0
    
    latest = window.iloc[-1]
    
    hr = latest.get('HR', 80)
    rr = latest.get('RR', 16)
    dbp = latest.get('DBP', 80)
    sbp = latest.get('SBP', 120)
    map_val = latest.get('MAP', 80)
    age = latest.get('age', 50)
    temp = latest.get('Temp', 37.0)
    
    score = 0.0
    
    # Tachycardia (strong indicator of cardiac stress)
    if hr >= 120:
        score += 0.4
    elif hr >= 110:
        score += 0.3
    elif hr <= 60:
        score += 0.2
    
    # Elevated respiratory rate (indicator of cardiac compensation)
    if rr >= 28:
        score += 0.4
    elif rr >= 24:
        score += 0.35
    elif rr >= 20:
        score += 0.3
    
    # Blood pressure instability is critical
    if sbp <= 100 or map_val <= 65:
        score += 0.35
    elif sbp <= 110 or map_val <= 75:
        score += 0.3
    
    # DBP component
    if dbp <= 50:
        score += 0.15
    
    # Age factor
    if age >= 75:
        score += 0.1
    elif age >= 65:
        score += 0.05
    
    # Only suppress if high fever suggests sepsis instead
    if temp >= 39.0:
        score *= 0.6
    
    return min(score, 1.0)

def respiratory_risk_score(window: pd.DataFrame) -> float:
    """Calculate respiratory failure risk score using ML model"""
    # Try ML model first
    ml_risk = _get_ml_prediction(window, 'respiratory')
    if ml_risk > 0:
        return ml_risk
    
    # Fallback to rule-based approach if ML model unavailable
    if window.empty:
        return 0.0
    
    latest = window.iloc[-1]
    
    spo2 = latest.get('SpO2', 98)
    rr = latest.get('RR', 16)
    hr = latest.get('HR', 80)
    
    score = 0.0
    
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
    
    # Tachypnea
    if rr >= 40:
        score += 0.3
    elif rr >= 35:
        score += 0.25
    elif rr >= 30:
        score += 0.2
    elif rr >= 25:
        score += 0.1
    
    # Compensatory tachycardia
    if hr >= 120 and spo2 <= 92:
        score += 0.2
    elif hr >= 100 and spo2 <= 90:
        score += 0.1
    
    # Trending SpO2 (if declining)
    if len(window) >= 5 and 'SpO2' in window.columns:
        spo2_values = window['SpO2'].fillna(98).values
        if len(spo2_values) >= 3:
            recent_trend = np.mean(spo2_values[-3:]) - np.mean(spo2_values[-6:-3]) if len(spo2_values) >= 6 else 0
            if recent_trend < -2:  # Declining SpO2
                score += 0.2
            elif recent_trend < -1:
                score += 0.1
    
    # Suppress if high fever suggests sepsis instead
    temp = latest.get('Temp', 37.0)
    if temp >= 38.5:
        score *= 0.8
    
    return min(score, 1.0)

def get_comprehensive_assessment(window: pd.DataFrame) -> Dict:
    """Get comprehensive patient risk assessment with proper class separation"""
    if window.empty:
        return {
            'overall_risk': 0.0,
            'primary_concern': 'Stable',
            'sepsis': {'risk': 0.0, 'prediction': 'Low Risk', 'confidence': 0.0, 'news2_score': 0, 'qsofa_score': 0},
            'cardiac': {'risk': 0.0, 'prediction': 'Low Risk', 'confidence': 0.0},
            'respiratory': {'risk': 0.0, 'prediction': 'Low Risk', 'confidence': 0.0},
            'recommendations': ['Continue routine monitoring'],
            'all_indicators': []
        }
    
    latest_vitals = window.iloc[-1]
    
    # Calculate clinical scores
    news2_score, news2_components = EarlyWarningScores.calculate_news2(latest_vitals)
    qsofa_score, qsofa_components = EarlyWarningScores.calculate_qsofa(latest_vitals)
    
    # Calculate disease-specific risk scores
    sepsis_risk = sepsis_risk_score(window)
    cardiac_risk = cardiac_risk_score(window)
    respiratory_risk = respiratory_risk_score(window)
    
    # Determine primary concern using argmax (highest risk)
    risks = {
        'sepsis': sepsis_risk,
        'cardiac': cardiac_risk,
        'respiratory': respiratory_risk
    }
    
    primary_concern_key = max(risks, key=risks.get)
    overall_risk = max(risks.values())
    
    # Map to readable primary concern
    concern_mapping = {
        'sepsis': 'Sepsis Risk',
        'cardiac': 'Cardiac Arrest Risk', 
        'respiratory': 'Respiratory Failure Risk'
    }
    
    if overall_risk < 0.3:
        primary_concern = 'Stable'
    else:
        primary_concern = concern_mapping[primary_concern_key]
    
    # Generate predictions for each category
    def get_prediction(risk_score):
        if risk_score >= 0.7:
            return 'High Risk'
        elif risk_score >= 0.4:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    # Build assessment dictionary
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
        'cardiac': {
            'risk': cardiac_risk,
            'prediction': get_prediction(cardiac_risk),
            'confidence': min(cardiac_risk + 0.1, 1.0),
            'cart_score': int(cardiac_risk * 20)  # Approximate CART score
        },
        'respiratory': {
            'risk': respiratory_risk,
            'prediction': get_prediction(respiratory_risk),
            'confidence': min(respiratory_risk + 0.1, 1.0),
            'raw_score': round(respiratory_risk * 10, 1)
        }
    }
    
    # Generate recommendations based on primary concern and risk level
    recommendations = []
    indicators = []
    
    if overall_risk >= 0.7:
        if primary_concern_key == 'sepsis':
            recommendations.extend([
                "🚨 IMMEDIATE: Initiate sepsis bundle protocol",
                "⚠️ Obtain blood cultures before antibiotics",
                "💉 Consider broad-spectrum antibiotics",
                "🩸 Check lactate and procalcitonin levels"
            ])
            indicators.extend(['High fever', 'Tachycardia', 'Hypotension'])
        elif primary_concern_key == 'cardiac':
            recommendations.extend([
                "🚨 IMMEDIATE: Prepare for possible cardiac arrest",
                "⚠️ Check ECG and cardiac enzymes",
                "💊 Review code status and crash cart availability",
                "🫀 Consider cardiology consultation"
            ])
            indicators.extend(['Severe tachycardia', 'Hypotension', 'High respiratory rate'])
        elif primary_concern_key == 'respiratory':
            recommendations.extend([
                "🚨 IMMEDIATE: Assess airway and breathing",
                "⚠️ Consider intubation if SpO₂ < 85%",
                "🫁 Increase oxygen support",
                "📊 Obtain arterial blood gas"
            ])
            indicators.extend(['Severe hypoxemia', 'Tachypnea', 'Respiratory distress'])
    elif overall_risk >= 0.4:
        recommendations.extend([
            f"⚠️ Monitor closely for {primary_concern.lower()}",
            "📈 Increase vital sign monitoring frequency",
            "🔍 Consider additional diagnostic tests"
        ])
    else:
        recommendations.append("✅ Continue routine monitoring")
    
    assessment['recommendations'] = recommendations
    assessment['all_indicators'] = indicators
    
    return assessment

def get_hard_alerts(vitals: pd.Series) -> List[str]:
    """Generate immediate hard alerts for critical vital signs"""
    alerts = []
    
    # Critical heart rate
    hr = vitals.get('HR', np.nan)
    if pd.notna(hr):
        if hr < 40:
            alerts.append('Severe Bradycardia (HR < 40)')
        elif hr > 150:
            alerts.append('Severe Tachycardia (HR > 150)')
    
    # Critical blood pressure
    sbp = vitals.get('SBP', np.nan)
    map_val = vitals.get('MAP', np.nan)
    if pd.notna(sbp) and sbp < 70:
        alerts.append('Severe Hypotension (SBP < 70)')
    if pd.notna(map_val) and map_val < 50:
        alerts.append('Critical MAP < 50')
    
    # Critical oxygenation
    spo2 = vitals.get('SpO2', np.nan)
    if pd.notna(spo2) and spo2 < 85:
        alerts.append('Severe Hypoxemia (SpO₂ < 85%)')
    
    # Critical respiratory rate
    rr = vitals.get('RR', np.nan)
    if pd.notna(rr):
        if rr < 8:
            alerts.append('Severe Bradypnea (RR < 8)')
        elif rr > 35:
            alerts.append('Severe Tachypnea (RR > 35)')
    
    # Critical temperature
    temp = vitals.get('Temp', np.nan)
    if pd.notna(temp):
        if temp < 35.0:
            alerts.append('Severe Hypothermia (< 35°C)')
        elif temp > 40.0:
            alerts.append('Severe Hyperthermia (> 40°C)')
    
    return alerts

def get_risk_level_info(risk_score: float) -> Dict[str, str]:
    """Get color and styling information for risk level"""
    if risk_score >= 0.7:
        return {
            'color': '🔴',
            'text_color': '#d32f2f',
            'bg_color': '#ffebee'
        }
    elif risk_score >= 0.4:
        return {
            'color': '🟡',
            'text_color': '#f57c00',
            'bg_color': '#fff3e0'
        }
    else:
        return {
            'color': '🟢',
            'text_color': '#388e3c',
            'bg_color': '#e8f5e8'
        }
