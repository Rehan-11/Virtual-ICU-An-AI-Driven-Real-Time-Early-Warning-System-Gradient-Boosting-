# app/ui_app.py - Enhanced Virtual ICU Monitor with Invigilator Controls
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.stream_pipeline import load_data, get_patient_ids, get_patient_df, get_window
from app.model import get_comprehensive_assessment, get_hard_alerts, get_risk_level_info, EarlyWarningScores

# Page configuration
st.set_page_config(
    page_title="Virtual ICU Monitor - AI Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with better visibility
st.markdown("""
<style>
.disease-risk-high {
    background-color: #d32f2f;
    color: white;
    border: 3px solid #b71c1c;
    border-radius: 10px;
    padding: 18px;
    margin: 12px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.disease-risk-medium {
    background-color: #f57c00;
    color: white;
    border: 3px solid #e65100;
    border-radius: 10px;
    padding: 18px;
    margin: 12px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.disease-risk-low {
    background-color: #388e3c;
    color: white;
    border: 3px solid #2e7d32;
    border-radius: 10px;
    padding: 18px;
    margin: 12px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.disease-risk-high h4, .disease-risk-medium h4, .disease-risk-low h4 {
    color: white !important;
    margin-top: 0;
    font-weight: bold;
}
.disease-risk-high h2, .disease-risk-medium h2, .disease-risk-low h2 {
    color: white !important;
    font-size: 2.5em;
    margin: 10px 0;
    font-weight: bold;
}
.disease-risk-high p, .disease-risk-medium p, .disease-risk-low p {
    color: white !important;
    font-weight: bold;
    margin: 8px 0;
}
.clinical-scores {
    background-color: rgba(255,255,255,0.2);
    border-radius: 6px;
    padding: 10px;
    margin: 8px 0;
}
.clinical-scores small {
    color: white !important;
    font-weight: bold;
}
.recommendation-high {
    background-color: #c62828;
    color: white;
    border: 3px solid #ad2121;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    font-weight: bold;
}
.recommendation-medium {
    background-color: #ef6c00;
    color: white;
    border: 3px solid #d84315;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    font-weight: bold;
}
.recommendation-low {
    background-color: #2e7d32;
    color: white;
    border: 3px solid #1b5e20;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    font-weight: bold;
}
.recommendation-high strong, .recommendation-medium strong, .recommendation-low strong {
    color: white !important;
    font-weight: bold;
}
.override-indicator {
    background-color: #ff5722;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
    margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'alerts_log' not in st.session_state:
    st.session_state.alerts_log = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'disease_alerts' not in st.session_state:
    st.session_state.disease_alerts = []
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None
if 'overrides' not in st.session_state:
    st.session_state.overrides = {}
if 'override_enabled' not in st.session_state:
    st.session_state.override_enabled = False

# Sidebar controls
st.sidebar.title("🏥 Virtual ICU AI Monitor")
st.sidebar.markdown("**Disease Prediction & Early Warning System**")

# File selection with enhanced data option
data_source = st.sidebar.selectbox("Data Source", 
    ["Enhanced Dataset (with disease patterns)", "Original Dataset"])

csv_path = "data/patient_vitals_enhanced.csv" if data_source.startswith("Enhanced") else "data/patient_vitals.csv"

# Load data
@st.cache_data(show_spinner=False)
def load_patient_data(csv_path):
    return load_data(csv_path)

try:
    df = load_patient_data(csv_path)
    if df.empty:
        st.error(f"No data found at {csv_path}")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Simple patient selection that always works
patients = get_patient_ids(df)

# Show patient info if available
if 'condition' in df.columns:
    st.sidebar.markdown("### 👥 Available Patients:")
    for p in patients:
        condition = df[df['patient_id'] == p]['condition'].iloc[0] if len(df[df['patient_id'] == p]) > 0 else 'unknown'
        age = df[df['patient_id'] == p]['age'].iloc if 'age' in df.columns else 'N/A'
        st.sidebar.write(f"**{p}**: {condition} (age {age})")

selected_patient = st.sidebar.selectbox("Select Patient", patients)

# Manual reset button for patient switching
if st.sidebar.button("🔄 Switch to This Patient", type="primary"):
    st.session_state.current_patient = selected_patient
    st.session_state.idx = 0
    st.session_state.alerts_log = []
    st.session_state.disease_alerts = []
    st.session_state.running = False
    # Clear overrides on patient switch
    st.session_state.overrides = {}
    st.session_state.override_enabled = False
    st.rerun()

# Set current patient if not set
if st.session_state.current_patient is None:
    st.session_state.current_patient = selected_patient

# Display current patient
st.sidebar.markdown("---")
st.sidebar.write(f"**🩺 Currently monitoring:** {st.session_state.current_patient}")
if selected_patient != st.session_state.current_patient:
    st.sidebar.warning("⚠️ Click 'Switch to This Patient' to change")

# Use the current patient from session state
active_patient = st.session_state.current_patient

# Control parameters
st.sidebar.markdown("### ⚙️ Monitoring Parameters")
window_minutes = st.sidebar.slider("Analysis Window (minutes)", 5, 60, 15)
update_speed = st.sidebar.slider("Simulation Speed (minutes/refresh)", 1, 15, 3)
risk_threshold = st.sidebar.slider("Alert Threshold", 0.1, 0.95, 0.6, 0.05)
refresh_interval = st.sidebar.slider("Refresh Rate (seconds)", 0.5, 3.0, 1.0, 0.1)

# AI Model Settings
st.sidebar.markdown("### 🤖 AI Prediction Settings")
enable_sepsis = st.sidebar.checkbox("Sepsis Prediction", True)
enable_cardiac = st.sidebar.checkbox("Cardiac Arrest Prediction", True)
enable_respiratory = st.sidebar.checkbox("Respiratory Failure Prediction", True)

# INVIGILATOR CONTROLS - NEW FEATURE
with st.sidebar.expander("🛠️ Invigilator Controls", expanded=False):
    st.markdown("**Live Vital Sign Override for Demonstration**")
    st.session_state.override_enabled = st.checkbox("Enable Live Override", value=st.session_state.override_enabled)
    st.caption("Adjust vitals below to simulate condition changes in real time. Overrides apply only to the current step (not saved to data).")
    
    if st.session_state.override_enabled:
        st.markdown("**Enter new values (leave blank to keep original):**")
        
        # Input widgets for overrides
        hr_ovr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, step=1, value=None, placeholder="e.g., 130")
        sbp_ovr = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=240, step=1, value=None, placeholder="e.g., 85")
        dbp_ovr = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=140, step=1, value=None, placeholder="e.g., 45")
        map_ovr = st.number_input("MAP (mmHg)", min_value=40, max_value=160, step=1, value=None, placeholder="auto from SBP/DBP")
        spo2_ovr = st.number_input("SpO₂ (%)", min_value=70, max_value=100, step=1, value=None, placeholder="e.g., 88")
        rr_ovr = st.number_input("Resp Rate (/min)", min_value=6, max_value=60, step=1, value=None, placeholder="e.g., 30")
        temp_ovr = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, step=0.1, value=None, placeholder="e.g., 39.5")
        
        col_apply, col_clear = st.columns(2)
        if col_apply.button("Apply Override", use_container_width=True, type="primary"):
            o = {}
            if hr_ovr is not None: o['HR'] = float(hr_ovr)
            if sbp_ovr is not None: o['SBP'] = float(sbp_ovr)
            if dbp_ovr is not None: o['DBP'] = float(dbp_ovr)
            if map_ovr is not None: o['MAP'] = float(map_ovr)
            if spo2_ovr is not None: o['SpO2'] = float(spo2_ovr)
            if rr_ovr is not None: o['RR'] = float(rr_ovr)
            if temp_ovr is not None: o['Temp'] = float(temp_ovr)
            st.session_state.overrides = o
            st.success("✅ Override applied for current step!")
            st.rerun()
        
        if col_clear.button("Reset Override", use_container_width=True):
            st.session_state.overrides = {}
            st.session_state.override_enabled = False
            st.info("Override cleared.")
            st.rerun()
        
        # Show current overrides
        if st.session_state.overrides:
            st.markdown("**Current Overrides:**")
            for k, v in st.session_state.overrides.items():
                st.write(f"• {k}: {v}")
    
    else:
        if st.session_state.overrides:
            st.caption("An override exists but is disabled. Enable to re-apply.")
        else:
            st.caption("No active overrides.")
    
    # Quick preset scenarios
    st.markdown("**Quick Test Scenarios:**")
    col_sep, col_card, col_resp = st.columns(3)
    
    if col_sep.button("🦠 Sepsis", help="Trigger sepsis-like vitals"):
        st.session_state.overrides = {'HR': 125, 'SBP': 88, 'DBP': 55, 'MAP': 66, 'SpO2': 92, 'RR': 28, 'Temp': 39.2}
        st.session_state.override_enabled = True
        st.rerun()
    
    if col_card.button("💓 Cardiac", help="Trigger cardiac arrest risk"):
        st.session_state.overrides = {'HR': 145, 'SBP': 75, 'DBP': 30, 'MAP': 45, 'SpO2': 88, 'RR': 32, 'Temp': 37.0}
        st.session_state.override_enabled = True
        st.rerun()
    
    if col_resp.button("🫁 Respiratory", help="Trigger respiratory failure"):
        st.session_state.overrides = {'HR': 115, 'SBP': 140, 'DBP': 90, 'MAP': 107, 'SpO2': 82, 'RR': 38, 'Temp': 37.1}
        st.session_state.override_enabled = True
        st.rerun()

# Playback controls
st.sidebar.markdown("### 🎮 Simulation Controls")
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("▶️", help="Start monitoring"):
    st.session_state.running = True
if col2.button("⏸️", help="Pause monitoring"):
    st.session_state.running = False
if col3.button("🔄", help="Reset to beginning"):
    st.session_state.idx = 0
    st.session_state.alerts_log = []
    st.session_state.disease_alerts = []
    st.session_state.running = False
    # Clear overrides on reset
    st.session_state.overrides = {}
    st.session_state.override_enabled = False

# Get patient data using the active patient
df_patient = get_patient_df(df, active_patient)
if df_patient.empty:
    st.error(f"No data found for patient {active_patient}")
    st.stop()

# Auto-advance if running
current_time = time.time()
if (st.session_state.running and 
    current_time - st.session_state.last_update >= refresh_interval):
    st.session_state.idx = min(st.session_state.idx + update_speed, len(df_patient) - 1)
    st.session_state.last_update = current_time

# Manual position control
manual_position = st.sidebar.slider("Manual Position (minutes from start)", 
    0, len(df_patient)-1, st.session_state.idx)
if manual_position != st.session_state.idx:
    st.session_state.idx = manual_position

# Get current window
window_seconds = window_minutes * 60
current_window = get_window(df_patient, st.session_state.idx, window_seconds)

# Apply overrides to the latest row if enabled
override_badge = ""
if st.session_state.override_enabled and st.session_state.overrides and not current_window.empty:
    cw = current_window.copy()
    latest_idx = cw.index[-1]
    for k, v in st.session_state.overrides.items():
        cw.loc[latest_idx, k] = v
    
    # If SBP/DBP set but MAP not set, recompute MAP
    sbp_set = ('SBP' in st.session_state.overrides)
    dbp_set = ('DBP' in st.session_state.overrides)
    map_set = ('MAP' in st.session_state.overrides)
    if sbp_set and dbp_set and not map_set and all(c in cw.columns for c in ['SBP','DBP']):
        cw.loc[latest_idx, 'MAP'] = (cw.loc[latest_idx, 'SBP'] + 2*cw.loc[latest_idx, 'DBP'])/3.0
    
    current_window = cw
    override_badge = " | 🛠️ OVERRIDE ACTIVE"

# Main interface
st.title("🏥 Virtual ICU Monitor - AI Disease Prediction System")

# Progress and time indicator
progress = (st.session_state.idx + 1) / len(df_patient)
current_time_sim = df_patient.iloc[st.session_state.idx]['timestamp'] if 'timestamp' in df_patient.columns else f"Minute {st.session_state.idx}"

col_prog1, col_prog2 = st.columns([3, 1])
with col_prog1:
    st.progress(progress, text=f"Patient {active_patient} | Progress: {progress:.1%} | Time: {current_time_sim}{override_badge}")
with col_prog2:
    status_color = "🟢" if st.session_state.running else "⏸️"
    st.markdown(f"**Status:** {status_color}")

if current_window.empty:
    st.warning("No data available for current analysis window")
    st.stop()

# Get comprehensive AI assessment
assessment = get_comprehensive_assessment(current_window)
latest_vitals = current_window.iloc[-1]

# Calculate clinical scores
news2_score, news2_components = EarlyWarningScores.calculate_news2(latest_vitals)
qsofa_score, qsofa_components = EarlyWarningScores.calculate_qsofa(latest_vitals)

# Current vitals display
st.subheader("📊 Current Vital Signs")
vital_col1, vital_col2, vital_col3, vital_col4, vital_col5, vital_col6 = st.columns(6)

with vital_col1:
    hr_val = latest_vitals.get('HR', np.nan)
    hr_display = f"{hr_val:.0f}" if pd.notna(hr_val) else "N/A"
    hr_color = "inverse" if pd.notna(hr_val) and (hr_val < 60 or hr_val > 100) else "normal"
    vital_col1.metric("Heart Rate", f"{hr_display} bpm", delta_color=hr_color)

with vital_col2:
    map_val = latest_vitals.get('MAP', np.nan)
    map_display = f"{map_val:.0f}" if pd.notna(map_val) else "N/A"
    vital_col2.metric("MAP", f"{map_display} mmHg")

with vital_col3:
    spo2_val = latest_vitals.get('SpO2', np.nan)
    spo2_display = f"{spo2_val:.0f}" if pd.notna(spo2_val) else "N/A"
    spo2_color = "inverse" if pd.notna(spo2_val) and spo2_val < 95 else "normal"
    vital_col3.metric("SpO₂", f"{spo2_display}%", delta_color=spo2_color)

with vital_col4:
    rr_val = latest_vitals.get('RR', np.nan)
    rr_display = f"{rr_val:.0f}" if pd.notna(rr_val) else "N/A"
    vital_col4.metric("Resp Rate", f"{rr_display} /min")

with vital_col5:
    temp_val = latest_vitals.get('Temp', np.nan)
    temp_display = f"{temp_val:.1f}" if pd.notna(temp_val) else "N/A"
    temp_color = "inverse" if pd.notna(temp_val) and (temp_val < 36 or temp_val > 38) else "normal"
    vital_col5.metric("Temperature", f"{temp_display}°C", delta_color=temp_color)

with vital_col6:
    risk_info = get_risk_level_info(assessment['overall_risk'])
    st.markdown(f"""
    <div style="background-color: {risk_info['bg_color']}; 
                padding: 1rem; 
                border-radius: 0.5rem; 
                text-align: center;
                border: 2px solid {risk_info['text_color']}">
        <h4 style="color: {risk_info['text_color']}; margin: 0;">
            {risk_info['color']} Risk: {assessment['overall_risk']:.2f}
        </h4>
        <p style="color: {risk_info['text_color']}; margin: 0; font-size: 0.8em;">
            {assessment['primary_concern']}
        </p>
    </div>
    """, unsafe_allow_html=True)

# AI DISEASE PREDICTIONS
st.subheader("🤖 AI Disease Risk Predictions")

pred_col1, pred_col2, pred_col3 = st.columns(3)

if enable_sepsis:
    with pred_col1:
        sepsis = assessment['sepsis']
        risk_class = "disease-risk-high" if sepsis['risk'] >= 0.7 else "disease-risk-medium" if sepsis['risk'] >= 0.4 else "disease-risk-low"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h4>🦠 Sepsis Risk</h4>
            <h2>{sepsis['risk']:.2f}</h2>
            <p><strong>{sepsis['prediction']}</strong></p>
            <div class="clinical-scores">
                <small>NEWS2: {sepsis['news2_score']} | qSOFA: {sepsis['qsofa_score']}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

if enable_cardiac:
    with pred_col2:
        cardiac = assessment['cardiac']
        risk_class = "disease-risk-high" if cardiac['risk'] >= 0.6 else "disease-risk-medium" if cardiac['risk'] >= 0.3 else "disease-risk-low"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h4>💓 Cardiac Arrest Risk</h4>
            <h2>{cardiac['risk']:.2f}</h2>
            <p><strong>{cardiac['prediction']}</strong></p>
            <div class="clinical-scores">
                <small>CART Score: {cardiac.get('cart_score', 0)}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

if enable_respiratory:
    with pred_col3:
        respiratory = assessment['respiratory']
        risk_class = "disease-risk-high" if respiratory['risk'] >= 0.7 else "disease-risk-medium" if respiratory['risk'] >= 0.4 else "disease-risk-low"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h4>🫁 Respiratory Failure Risk</h4>
            <h2>{respiratory['risk']:.2f}</h2>
            <p><strong>{respiratory['prediction']}</strong></p>
            <div class="clinical-scores">
                <small>Raw Score: {respiratory.get('raw_score', 0)}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ALERTS MANAGEMENT
max_risk = assessment['overall_risk']
if max_risk >= risk_threshold:
    alert_time = datetime.now().strftime("%H:%M:%S")
    new_alert = {
        'time': alert_time,
        'primary_concern': assessment['primary_concern'],
        'risk_level': max_risk,
        'indicators': assessment['all_indicators'][:3]
    }
    
    if not st.session_state.disease_alerts or st.session_state.disease_alerts[-1]['time'] != alert_time:
        st.session_state.disease_alerts.append(new_alert)

# Display immediate alerts
hard_alerts = get_hard_alerts(latest_vitals)
if hard_alerts or max_risk >= risk_threshold:
    st.subheader("🚨 IMMEDIATE ALERTS")
    
    if hard_alerts:
        for alert in hard_alerts:
            st.markdown(f"""
            <div class="recommendation-high">
                <strong>⚠️ CRITICAL: {alert}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    if max_risk >= risk_threshold:
        st.markdown(f"""
        <div class="recommendation-high">
            <strong>🤖 AI ALERT: {assessment['primary_concern']}</strong><br>
            Risk Level: {max_risk:.2f} | Confidence: {assessment['sepsis']['confidence']:.2f}
        </div>
        """, unsafe_allow_html=True)

# VITAL SIGNS CHARTS
st.subheader("📈 Vital Signs Monitoring")

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=("Heart Rate (bpm)", "Blood Pressure (mmHg)", 
                   "Oxygen Saturation (%)", "Respiratory Rate (/min)",
                   "Temperature (°C)", "Risk Trends"),
    vertical_spacing=0.08,
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

if 'timestamp' in current_window.columns:
    x_axis = pd.to_datetime(current_window['timestamp'])
else:
    x_axis = list(range(len(current_window)))

# Heart Rate
if 'HR' in current_window.columns:
    fig.add_trace(
        go.Scatter(x=x_axis, y=current_window['HR'], 
                  mode='lines+markers', name='HR',
                  line=dict(color='#e74c3c', width=2),
                  marker=dict(size=4)),
        row=1, col=1
    )
    fig.add_hline(y=100, line_dash="dash", line_color="orange", row=1, col=1)
    fig.add_hline(y=60, line_dash="dash", line_color="orange", row=1, col=1)

# Blood Pressure
if all(col in current_window.columns for col in ['SBP', 'DBP', 'MAP']):
    fig.add_trace(
        go.Scatter(x=x_axis, y=current_window['SBP'], 
                  mode='lines', name='SBP',
                  line=dict(color='#3498db', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=current_window['DBP'], 
                  mode='lines', name='DBP',
                  line=dict(color='#2980b9', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=current_window['MAP'], 
                  mode='lines+markers', name='MAP',
                  line=dict(color='#1f4e79', width=3),
                  marker=dict(size=4)),
        row=1, col=2
    )
    fig.add_hline(y=65, line_dash="dash", line_color="red", row=1, col=2)

# SpO2
if 'SpO2' in current_window.columns:
    fig.add_trace(
        go.Scatter(x=x_axis, y=current_window['SpO2'], 
                  mode='lines+markers', name='SpO₂',
                  line=dict(color='#27ae60', width=2),
                  marker=dict(size=4)),
        row=2, col=1
    )
    fig.add_hline(y=95, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=90, line_dash="dash", line_color="red", row=2, col=1)

# Respiratory Rate
if 'RR' in current_window.columns:
    fig.add_trace(
        go.Scatter(x=x_axis, y=current_window['RR'], 
                  mode='lines+markers', name='RR',
                  line=dict(color='#8e44ad', width=2),
                  marker=dict(size=4)),
        row=2, col=2
    )
    fig.add_hline(y=20, line_dash="dash", line_color="orange", row=2, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=2)

# Temperature
if 'Temp' in current_window.columns:
    fig.add_trace(
        go.Scatter(x=x_axis, y=current_window['Temp'], 
                  mode='lines+markers', name='Temperature',
                  line=dict(color='#f39c12', width=2),
                  marker=dict(size=4)),
        row=3, col=1
    )
    fig.add_hline(y=38.0, line_dash="dash", line_color="orange", row=3, col=1)
    fig.add_hline(y=36.0, line_dash="dash", line_color="orange", row=3, col=1)

# Risk trends
risk_timeline = []
for i in range(max(0, st.session_state.idx - 30), st.session_state.idx + 1):
    if i < len(df_patient):
        temp_window = get_window(df_patient, i, window_seconds)
        if not temp_window.empty:
            temp_assessment = get_comprehensive_assessment(temp_window)
            risk_timeline.append(temp_assessment['overall_risk'])

if risk_timeline:
    # FIX: Convert range to list here
    risk_x = x_axis[-len(risk_timeline):] if len(x_axis) >= len(risk_timeline) else list(range(len(risk_timeline)))
    fig.add_trace(
        go.Scatter(x=risk_x, y=risk_timeline,
                  mode='lines+markers', name='Overall Risk',
                  line=dict(color='#e74c3c', width=3),
                  marker=dict(size=6)),
        row=3, col=2
    )
    fig.add_hline(y=risk_threshold, line_dash="dash", line_color="red", row=3, col=2)

fig.update_layout(height=700, showlegend=False, title_text="Real-time Vital Signs & AI Risk Assessment")
fig.update_yaxes(range=[30, 200], row=1, col=1)
fig.update_yaxes(range=[40, 200], row=1, col=2)
fig.update_yaxes(range=[70, 100], row=2, col=1)
fig.update_yaxes(range=[5, 50], row=2, col=2)
fig.update_yaxes(range=[35, 42], row=3, col=1)
fig.update_yaxes(range=[0, 1], row=3, col=2)

st.plotly_chart(fig, use_container_width=True)

# CLINICAL DECISION SUPPORT
st.subheader("🩺 Clinical Decision Support")

cds_col1, cds_col2 = st.columns([2, 1])

with cds_col1:
    st.markdown("### 📋 AI Recommendations")
    recommendations = assessment['recommendations']
    
    for rec in recommendations:
        if "IMMEDIATE" in rec or "🚨" in rec:
            rec_class = "recommendation-high"
        elif "⚠️" in rec or "Consider" in rec:
            rec_class = "recommendation-medium"
        else:
            rec_class = "recommendation-low"
        
        st.markdown(f"""
        <div class="{rec_class}">
            {rec}
        </div>
        """, unsafe_allow_html=True)

with cds_col2:
    st.markdown("### 📊 Clinical Scores")
    st.markdown(f"""
    <div class="clinical-scores">
        <strong>NEWS2 Score:</strong> {news2_score}<br>
        <strong>qSOFA Score:</strong> {qsofa_score}<br>
        <strong>Risk Level:</strong> {assessment['primary_concern']}<br>
        <strong>Confidence:</strong> {assessment['sepsis']['confidence']:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Score Breakdown"):
        st.write("**NEWS2 Components:**")
        for component, score in news2_components.items():
            st.write(f"- {component}: {score}")
        
        st.write("**qSOFA Components:**")
        for component, score in qsofa_components.items():
            st.write(f"- {component}: {score}")

# Patient details
with st.expander("👤 Patient Details & Analytics"):
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.write(f"**Patient ID:** {active_patient}")
        if 'condition' in df_patient.columns:
            st.write(f"**Condition:** {df_patient['condition'].iloc[0]}")
        if 'age' in df_patient.columns:
            st.write(f"**Age:** {df_patient['age'].iloc}")
        st.write(f"**Current Time:** {current_time_sim}")
        st.write(f"**Window Size:** {len(current_window)} data points")
    
    with detail_col2:
        st.write("**Vital Signs Summary (Current Window):**")
        if not current_window.empty:
            summary_data = []
            for vital in ['HR', 'MAP', 'SpO2', 'RR', 'Temp']:
                if vital in current_window.columns:
                    values = current_window[vital].dropna()
                    if not values.empty:
                        trend_slope = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                        summary_data.append({
                            'Vital': vital,
                            'Current': f"{values.iloc[-1]:.1f}",
                            'Min': f"{values.min():.1f}",
                            'Max': f"{values.max():.1f}",
                            'Trend': f"{trend_slope:+.2f}/min"
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

# Auto-refresh mechanism
if st.session_state.running and st.session_state.idx < len(df_patient) - 1:
    time.sleep(refresh_interval)
    st.rerun()
elif st.session_state.running and st.session_state.idx >= len(df_patient) - 1:
    st.session_state.running = False
    st.success("✅ Simulation completed! Use Reset button to restart.")

# Footer
st.markdown("---")
st.markdown("**Virtual ICU Monitor v2.0** - AI-powered early disease detection and prediction system | "
           "Based on NEWS2, qSOFA, and CART clinical scoring systems")
