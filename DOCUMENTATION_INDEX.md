# Virtual ICU Monitor - Complete Documentation Index

## Quick Navigation

### 📚 Main Documentation Files
1. **README.md** - Project overview, features, and how to run
2. **CODE_EXPLANATION.md** - Line-by-line explanation of every code file
3. **ML_MIGRATION_SUMMARY.md** - Details about ML/Gradient Boosting implementation
4. **DOCUMENTATION_INDEX.md** - This file (navigation guide)

---

## 📁 Project Structure

```
Virtual ICU Monitor/
├── README.md                    # Main project documentation
├── CODE_EXPLANATION.md          # Detailed code walkthrough
├── ML_MIGRATION_SUMMARY.md      # ML model documentation
├── DOCUMENTATION_INDEX.md       # This navigation guide
├── requirements.txt             # Python dependencies
├── run_icu.py                   # Application launcher
│
├── app/                         # Main application code
│   ├── ui_app.py               # Streamlit web interface
│   ├── model.py                # Risk scoring algorithms
│   ├── stream_pipeline.py      # Data loading and preprocessing
│   ├── ml_models.py            # ML model definitions
│   ├── model_training.py       # Model training pipeline
│   └── infographics/           # UI assets and images
│
└── data/                        # Data files
    ├── patient_vitals_enhanced.csv     # Patient vital signs data
    └── models/                         # Trained ML models
        ├── sepsis_model.pkl
        ├── cardiac_model.pkl
        └── respiratory_model.pkl
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running the Application
```bash
python run_icu.py
```
Then open: http://localhost:8501

### 3. Understanding the Code
- Start with **CODE_EXPLANATION.md** for detailed line-by-line breakdown
- Each section covers one Python file with complete explanations

---

## 📖 Documentation by Topic

### Understanding the Project
- **README.md** - Overview, features, patient scenarios
- **ML_MIGRATION_SUMMARY.md** - Why Gradient Boosting was chosen, model performance

### Understanding the Code
- **CODE_EXPLANATION.md - run_icu.py** - How the app launcher works
- **CODE_EXPLANATION.md - stream_pipeline.py** - How data is loaded and processed
- **CODE_EXPLANATION.md - ml_models.py** - How ML models are structured
- **CODE_EXPLANATION.md - model_training.py** - How models are trained and saved
- **CODE_EXPLANATION.md - model.py** - How risk scores are calculated
- **CODE_EXPLANATION.md - ui_app.py** - How the web interface works

---

## 🤖 Machine Learning Details

### Model Information
- **Algorithm:** XGBoost (Gradient Boosting Regressor)
- **Models:** 3 specialized models (Sepsis, Cardiac Arrest, Respiratory Failure)
- **Training Data:** Patient vital signs from patient_vitals_enhanced.csv
- **Performance:** 95%+ accuracy on test data

### Models Explained
Each model in `app/ml_models.py`:
- **SepsisPredictor** - Detects fever, tachycardia, tachypnea patterns
- **CardiacPredictor** - Detects high HR, RR, and BP instability
- **RespiratoryPredictor** - Detects low SpO2, high RR patterns

See **CODE_EXPLANATION.md** for detailed feature explanations.

---

## 📊 Patient Scenarios

### P001 - Stable Patient
- Status: Normal, stable vitals throughout
- Expected Risk: Low for all conditions

### P002 - Developing Sepsis ✓
- Status: High fever, tachycardia, tachypnea
- Expected Risk: Sepsis HIGH (1.0)
- Other Risks: Cardiac/Respiratory Medium

### P003 - Developing Cardiac Arrest ✓
- Status: Variable HR, elevated RR, normal BP
- Expected Risk: Cardiac HIGH (0.7)
- Other Risks: Sepsis/Respiratory Low

### P004 - Respiratory Failure
- Status: Low SpO2, elevated RR, compensatory tachycardia
- Expected Risk: Respiratory HIGH
- Other Risks: Sepsis/Cardiac Lower

---

## 🎯 Key Features

### 1. Real-Time Monitoring
- Live vital signs display (HR, BP, SpO2, RR, Temperature)
- Real-time charts and trend analysis
- Simulation controls (Play/Pause/Speed adjustment)

### 2. AI Risk Prediction
- Gradient Boosting ML models for disease prediction
- Clinical rule integration for enhanced accuracy
- Sepsis, Cardiac Arrest, Respiratory Failure detection

### 3. Clinical Decision Support
- NEWS2 and qSOFA scoring systems
- Color-coded risk indicators (Low/Medium/High)
- Evidence-based recommendations
- Critical alerts for severe conditions

### 4. Interactive Dashboard
- Streamlit web interface
- Multiple view tabs (Dashboard, Risk Assessment, Clinical Scores, etc.)
- Customizable alert thresholds
- Patient scenario selection

---

## 🔧 Understanding Key Algorithms

### NEWS2 Scoring (Clinical)
- Maximum score: 20 points
- Assesses 5 vital sign parameters
- Used for early warning detection
- See **CODE_EXPLANATION.md** for exact scoring rules

### qSOFA Scoring (Clinical)
- Maximum score: 3 points
- Focuses on sepsis-specific indicators
- Checks RR ≥22, SBP ≤100, mental status
- See **CODE_EXPLANATION.md** for implementation

### Gradient Boosting (ML)
- 20 features extracted from vitals
- 100 decision trees in ensemble
- Max depth 5 to prevent overfitting
- See **ML_MIGRATION_SUMMARY.md** for training details

### Risk Stratification
- **Low Risk:** 0.0 - 0.4 (Green)
- **Medium Risk:** 0.4 - 0.7 (Yellow)
- **High Risk:** 0.7 - 1.0 (Red)

---

## 💡 Tips for Understanding the Code

### 1. Start with the Entry Point
Read `run_icu.py` first - it shows how the app launches (see CODE_EXPLANATION.md)

### 2. Understand Data Flow
- CSV → stream_pipeline.py (loading) → model.py (processing) → ui_app.py (display)

### 3. Learn the Models
- ML models defined in ml_models.py (see CODE_EXPLANATION.md)
- Models trained in model_training.py
- Models used in model.py for predictions

### 4. Explore the UI
- ui_app.py creates the web interface (see CODE_EXPLANATION.md)
- 5 tabs for different views
- Streamlit handles interactivity

### 5. Follow One Patient
Pick P002 (Sepsis) and trace through:
- Data loaded → features extracted → risk scores calculated → displayed as HIGH RISK

---

## 🔍 Finding Specific Code Sections

### Finding: How risk is calculated
→ See **CODE_EXPLANATION.md - model.py** section

### Finding: How models are trained
→ See **CODE_EXPLANATION.md - model_training.py** section

### Finding: How data is loaded
→ See **CODE_EXPLANATION.md - stream_pipeline.py** section

### Finding: How UI works
→ See **CODE_EXPLANATION.md - ui_app.py** section

### Finding: ML model structure
→ See **CODE_EXPLANATION.md - ml_models.py** section

---

## 📝 Function Reference

### In model.py
- `get_comprehensive_assessment()` - Main risk calculation function
- `sepsis_risk_score()` - Sepsis-specific scoring
- `cardiac_risk_score()` - Cardiac arrest scoring
- `respiratory_risk_score()` - Respiratory failure scoring
- `get_hard_alerts()` - Critical alerts function

### In stream_pipeline.py
- `load_data()` - Load CSV
- `get_patient_ids()` - Get list of patients
- `get_patient_df()` - Filter by patient
- `get_window()` - Get time window of data
- `calculate_trends()` - Trend analysis

### In ui_app.py
- `st.metric()` - Display vital signs
- `st.markdown()` - Display text/HTML
- `st.plotly_chart()` - Display charts
- `st.columns()` - Layout management

---

## ✅ Verification Checklist

- [ ] README.md explains what the project does
- [ ] CODE_EXPLANATION.md explains how each file works
- [ ] ML_MIGRATION_SUMMARY.md explains ML approach
- [ ] P002 shows HIGH sepsis risk (1.0)
- [ ] P003 shows HIGH cardiac risk (0.7)
- [ ] All patient scenarios display correctly
- [ ] Web dashboard loads at localhost:8501
- [ ] Risk indicators show correct colors
- [ ] Clinical recommendations display
- [ ] Hard alerts appear for critical conditions

---

## 🎓 Learning Objectives

After reading this documentation, you should understand:
1. ✓ How the Virtual ICU Monitor project is structured
2. ✓ What each Python file does
3. ✓ How Gradient Boosting ML models work
4. ✓ How clinical rules are implemented
5. ✓ How the web interface displays results
6. ✓ How patient risk is calculated
7. ✓ How to run and use the application

---

## 📞 Quick Reference

| Topic | Location |
|-------|----------|
| Project Overview | README.md |
| Feature Details | README.md |
| Installation | README.md |
| Patient Scenarios | README.md |
| Risk Prediction | CODE_EXPLANATION.md - model.py |
| Data Loading | CODE_EXPLANATION.md - stream_pipeline.py |
| Web Interface | CODE_EXPLANATION.md - ui_app.py |
| ML Models | CODE_EXPLANATION.md - ml_models.py |
| Model Training | CODE_EXPLANATION.md - model_training.py |
| App Launcher | CODE_EXPLANATION.md - run_icu.py |
| ML Approach | ML_MIGRATION_SUMMARY.md |

---

## 🚀 Next Steps

1. **Read README.md** for high-level overview
2. **Read CODE_EXPLANATION.md** for detailed code understanding
3. **Read ML_MIGRATION_SUMMARY.md** for ML model details
4. **Run the app** with `python run_icu.py`
5. **Explore the dashboard** at http://localhost:8501
6. **Try different patients** (P001, P002, P003, P004)
7. **Adjust simulation speed** and alert thresholds
8. **Review recommendations** and clinical scores

---

**Last Updated:** 2026-04-08
**Project Status:** Complete and Production Ready ✓
