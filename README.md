# SmartCare Hospital - No-Show Prediction

## Project Overview
This project predicts whether a patient will miss their hospital appointment using machine learning.

## Dataset
- 1000 patient records
- 33 features
- Target: no_show (1 = missed, 0 = attended)

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- Naive Bayes
- KNN
- XGBoost

## Best Model
Random Forest (Tuned) - F1 Score: 0.60

## Key Features
- waiting_days
- long_wait_flag
- appointment_dayofweek
- previous_appointments
- missed_ratio

## Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP (Explainable AI)
- Streamlit (Prototype)

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
