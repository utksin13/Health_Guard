
import streamlit as st
import pickle
import numpy as np

# Load the models
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_model.sav', 'rb'))

def predict_parkinsons(features):
    features = np.array(features).reshape(1, -1)
    return parkinsons_model.predict(features)[0]

def predict_diabetes(features):
    features = np.array(features).reshape(1, -1)
    return diabetes_model.predict(features)[0]

def predict_heart(features):
    features = np.array(features).reshape(1, -1)
    return heart_model.predict(features)[0]

# Navigation pane
st.sidebar.title("Health Prediction Models")
app_mode = st.sidebar.selectbox("Choose the Prediction Model", 
                                ["Parkinson's Prediction", "Diabetes Prediction", "Heart Disease Prediction"])

if app_mode == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")

    # Parkinson's Model Input Fields
    fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=300.0, step=0.1)
    fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=300.0, step=0.1)
    flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=300.0, step=0.1)
    jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, step=0.01)
    jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=0.1, step=0.001)
    rap = st.number_input('MDVP:RAP', min_value=0.0, max_value=1.0, step=0.01)
    ppq = st.number_input('MDVP:PPQ', min_value=0.0, max_value=1.0, step=0.01)
    ddp = st.number_input('Jitter:DDP', min_value=0.0, max_value=3.0, step=0.1)
    shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, step=0.01)
    shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=2.0, step=0.1)
    apq3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=1.0, step=0.01)
    apq5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=1.0, step=0.01)
    apq = st.number_input('MDVP:APQ', min_value=0.0, max_value=1.0, step=0.01)
    dda = st.number_input('Shimmer:DDA', min_value=0.0, max_value=3.0, step=0.1)
    nhr = st.number_input('NHR', min_value=0.0, max_value=1.0, step=0.01)
    hnr = st.number_input('HNR', min_value=0.0, max_value=50.0, step=0.1)
    rpde = st.number_input('RPDE', min_value=0.0, max_value=1.0, step=0.01)
    dfa = st.number_input('DFA', min_value=0.0, max_value=1.0, step=0.01)
    spread1 = st.number_input('spread1', min_value=-10.0, max_value=0.0, step=0.1)
    spread2 = st.number_input('spread2', min_value=0.0, max_value=1.0, step=0.01)
    d2 = st.number_input('D2', min_value=0.0, max_value=5.0, step=0.1)
    ppe = st.number_input('PPE', min_value=0.0, max_value=1.0, step=0.01)

    if st.button('Predict Parkinson\'s'):
        features = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        result = predict_parkinsons(features)
        st.success(f"Parkinson's Prediction: {'Positive' if result == 1 else 'Negative'}")

elif app_mode == "Diabetes Prediction":
    st.title("Diabetes Prediction")

    # Diabetes Model Input Fields
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose', min_value=0.0, max_value=200.0, step=0.1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0.0, max_value=200.0, step=0.1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0.0, max_value=100.0, step=0.1)
    insulin = st.number_input('Insulin', min_value=0.0, max_value=900.0, step=0.1)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input('Age', min_value=0, max_value=120, step=1)

    if st.button('Predict Diabetes'):
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_diabetes(features)
        st.success(f"Diabetes Prediction: {'Positive' if result == 1 else 'Negative'}")

elif app_mode == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")

    # Heart Disease Model Input Fields
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    sex = st.number_input('Sex (0: Female, 1: Male)', min_value=0, max_value=1, step=1)
    cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, step=1)
    trestbps = st.number_input('Resting Blood Pressure', min_value=0.0, max_value=200.0, step=0.1)
    chol = st.number_input('Serum Cholestoral', min_value=0.0, max_value=600.0, step=0.1)
    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)', min_value=0, max_value=1, step=1)
    restecg = st.number_input('Resting ECG (0, 1, 2)', min_value=0, max_value=2, step=1)
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0.0, max_value=250.0, step=0.1)
    exang = st.number_input('Exercise Induced Angina (1 = Yes; 0 = No)', min_value=0, max_value=1, step=1)
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, step=0.1)
    slope = st.number_input('Slope of the Peak Exercise ST Segment', min_value=0, max_value=2, step=1)
    ca = st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, step=1)
    thal = st.number_input('Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)', min_value=1, max_value=3, step=1)

    if st.button('Predict Heart Disease'):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = predict_heart(features)
        st.success(f"Heart Disease Prediction: {'Positive' if result == 1 else 'Negative'}")
