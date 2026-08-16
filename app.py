import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="SmartCare No-Show Predictor", page_icon="🏥", layout="centered")

@st.cache_resource
def load_model():
    # Try both possible filenames
    possible_paths = [
        'best_model_random_forest.pkl',
        'best_model_random_forest .pkl',
        './best_model_random_forest.pkl',
    ]
    
    model = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.success(f"✅ Model loaded!")
                break
            except:
                pass
    
    if model is None:
        st.error("❌ Model file not found!")
        st.info(f"📁 Files: {os.listdir('.')}")
        return None, None
    
    # Load feature info
    feature_columns = None
    if os.path.exists('feature_info.json'):
        with open('feature_info.json', 'r') as f:
            feature_info = json.load(f)
            feature_columns = feature_info.get('columns', [])
    
    if feature_columns is None:
        if os.path.exists('X_processed.csv'):
            X = pd.read_csv('X_processed.csv')
            feature_columns = X.columns.tolist()
    
    if feature_columns is None:
        st.error("❌ Feature columns not found!")
        return None, None
    
    return model, feature_columns

model, feature_columns = load_model()

if model is None or feature_columns is None:
    st.error("⚠️ Could not load model. Please check files.")
    st.stop()

st.title("🏥 SmartCare Hospital")
st.subheader("Appointment No-Show Risk Predictor")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        department = st.selectbox("Department", ["General Medicine", "Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Radiology"])
        diagnosis = st.selectbox("Diagnosis", ["Migraine", "Diabetes", "Back Pain", "Asthma", "Hypertension", "Fracture"])
    with col2:
        waiting_days = st.slider("Waiting Days", 0, 60, 20)
        previous_appointments = st.number_input("Previous Appointments", 0, 20, 3)
        missed_prev = st.number_input("Missed Previous", 0, 10, 0)
        previous_admissions = st.number_input("Previous Admissions", 0, 10, 0)
        payment_method = st.selectbox("Payment Method", ["Cash", "Card", "Online", "Insurance"])
        consultation_fee = st.number_input("Consultation Fee (LKR)", 1500, 4000, 2000)
    submitted = st.form_submit_button("🔮 Predict No-Show Risk")

if submitted:
    row = {c: 0 for c in feature_columns}
    row.update({
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "waiting_days": waiting_days,
        "previous_appointments": previous_appointments,
        "missed_previous_appointments": missed_prev,
        "previous_admissions": previous_admissions,
        "consultation_fee_lkr": consultation_fee,
        "appointment_month": 6,
        "appointment_dayofweek": 3,
        "is_weekend_appt": 0,
        "missed_ratio": missed_prev / (previous_appointments + 1) if previous_appointments > 0 else 0,
        "is_first_visit": int(previous_appointments == 0),
        "long_wait_flag": int(waiting_days > 22)
    })
    for prefix, val in [("blood_group_", blood_group), ("department_", department), ("diagnosis_", diagnosis), ("payment_method_", payment_method)]:
        key = f"{prefix}{val}"
        if key in row:
            row[key] = 1
    if age <= 18: ag = "0-18"
    elif age <= 35: ag = "19-35"
    elif age <= 50: ag = "36-50"
    elif age <= 65: ag = "51-65"
    else: ag = "65+"
    key = f"age_group_{ag}"
    if key in row:
        row[key] = 1
    
    X_input = pd.DataFrame([row])[feature_columns]
    proba = model.predict_proba(X_input)[0, 1]
    pred = int(proba >= 0.5)
    
    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ HIGH RISK - No-Show Probability: {proba*100:.1f}%")
    else:
        st.success(f"✅ Likely to Attend - No-Show Probability: {proba*100:.1f}%")
    st.progress(proba)
    
    # ============================================
    # SHAP EXPLANATION - FIXED VERSION
    # ============================================
    st.markdown("#### Why this prediction?")
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        
        # Debug info (remove these lines later if you want)
        st.write(f"SHAP type: {type(shap_values)}")
        if isinstance(shap_values, list):
            st.write(f"List length: {len(shap_values)}")
        
        # Try to extract SHAP values correctly
        if isinstance(shap_values, list):
            # For binary classification, use class 1 (no-show)
            if len(shap_values) == 2:
                sv = np.array(shap_values[1])  # Class 1
            else:
                sv = np.array(shap_values[0])
        else:
            sv = np.array(shap_values)
        
        # Handle 3D arrays [samples, features, classes]
        if sv.ndim == 3:
            sv = sv[0, :, 1] if sv.shape[2] > 1 else sv[0, :, 0]
        elif sv.ndim == 2:
            sv = sv[0]  # First sample
        
        # Flatten if needed
        if sv.ndim > 1:
            sv = sv.flatten()
        
        # Match lengths
        if len(sv) != len(feature_columns):
            st.warning(f"Length mismatch: SHAP values ({len(sv)}) vs features ({len(feature_columns)})")
            min_len = min(len(sv), len(feature_columns))
            sv = sv[:min_len]
            cols = feature_columns[:min_len]
        else:
            cols = feature_columns
        
        # Get top 5
        contrib = pd.Series(sv, index=cols)
        contrib = contrib.sort_values(key=abs, ascending=False).head(5)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#DD4C4C' if v > 0 else '#4C72B0' for v in contrib.values]
        ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
        ax.set_xlabel("SHAP value (impact on no-show risk)")
        ax.set_title("Top factors for this patient")
        st.pyplot(fig)
        st.caption("🔴 Red increases risk | 🔵 Blue decreases risk")
        
    except Exception as e:
        st.warning(f"SHAP note: {str(e)[:100]}")
        # Fallback: Show feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("**Top features (model importance):**")
            importance = pd.Series(model.feature_importances_, index=feature_columns)
            top_features = importance.sort_values(ascending=False).head(5)
            for feat, imp in top_features.items():
                st.write(f"- {feat}: {imp:.3f}")

st.markdown("---")
st.caption("SmartCare Hospital AI System | CCS3440 Coursework Prototype")