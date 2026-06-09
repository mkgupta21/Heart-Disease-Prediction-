import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")

age = st.slider("Age", 20, 80)
sex = st.selectbox("Sex", ["Female","Male"])
chest_pain = st.selectbox("Chest Pain Type", [1,2,3,4])
bp = st.slider("Resting Blood Pressure", 90, 200)
chol = st.slider("Cholesterol", 100, 400)
max_hr = st.slider("Max Heart Rate", 70, 200)

thal = st.selectbox("Thal", ["Normal","Fixed Defect","Reversible Defect"])

sex = 1 if sex == "Male" else 0

thal_fixed = 1 if thal == "Fixed Defect" else 0
thal_normal = 1 if thal == "Normal" else 0
thal_reversible = 1 if thal == "Reversible Defect" else 0

if st.button("Predict"):

    sample = pd.DataFrame([[1,bp,chest_pain,0,0,0,chol,0.5,sex,age,max_hr,0,
                            thal_fixed,thal_normal,thal_reversible]],
    columns=['slope_of_peak_exercise_st_segment',
             'resting_blood_pressure',
             'chest_pain_type',
             'num_major_vessels',
             'fasting_blood_sugar_gt_120_mg_per_dl',
             'resting_ekg_results',
             'serum_cholesterol_mg_per_dl',
             'oldpeak_eq_st_depression',
             'sex',
             'age',
             'max_heart_rate_achieved',
             'exercise_induced_angina',
             'thal_fixed_defect',
             'thal_normal',
             'thal_reversible_defect'])

    cols_to_scale = ['age','resting_blood_pressure',
                     'serum_cholesterol_mg_per_dl',
                     'max_heart_rate_achieved',
                     'oldpeak_eq_st_depression']

    sample[cols_to_scale] = scaler.transform(sample[cols_to_scale])

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")
