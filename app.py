import streamlit as st
import pickle
import numpy as np
import os
from streamlit_option_menu import option_menu

# Page Configurations
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Load models
def load_model(filename):
    if not os.path.exists(filename):
        st.error(f"❌ Model file '{filename}' not found! Please check your repository.")
        st.stop()
    return pickle.load(open(filename, 'rb'))

models = {
    'diabetes': load_model("best_diabetes_model.sav"),
    'heart_disease': load_model("heart_disease_model.sav"),
    'parkinsons': load_model("parkinsons_model.sav"),
    'lung_cancer': load_model("lungs_disease_model.sav"),
    'thyroid': load_model("Thyroid_model.sav")
}

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Diabetes", "Heart Disease", "Parkinson's", "Lung Cancer", "Thyroid"],
        icons=["activity", "heart", "person", "lungs", "droplet"],
        menu_icon="hospital",
        default_index=0,
    )

# Function to take user input
def user_input(label, key, type="number"):
    if type == "toggle":
        return int(st.toggle(label, key=key))
    return float(st.number_input(label, step=1.0, key=key))

# Function to make predictions
def make_prediction(model, features):
    try:
        input_data = np.array(features).reshape(1, -1)
        if len(features) != model.n_features_in_:
            st.error(f"❌ Expected {model.n_features_in_} features, but got {len(features)}!")
            return None
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Diabetes Prediction
if selected == "Diabetes":
    st.header("Diabetes Prediction")
    features = [
        user_input("Pregnancies", "Pregnancies"),
        user_input("Glucose Level", "Glucose"),
        user_input("Blood Pressure", "BloodPressure"),
        user_input("Skin Thickness", "SkinThickness"),
        user_input("Insulin", "Insulin"),
        user_input("BMI", "BMI"),
        user_input("Diabetes Pedigree Function", "DiabetesPedigreeFunction"),
        user_input("Age", "Age")
    ]
    if st.button("Check Diabetes"):
        result = make_prediction(models['diabetes'], features)
        st.success("Diabetic" if result == 1 else "Not Diabetic")

# Heart Disease Prediction
elif selected == "Heart Disease":
    st.header("Heart Disease Prediction")
    features = [
        user_input("Age", "age"),
        user_input("Sex (1=Male, 0=Female)", "sex", "toggle"),
        user_input("Chest Pain Type (0-3)", "cp"),
        user_input("Resting Blood Pressure", "trestbps"),
        user_input("Serum Cholesterol", "chol"),
        user_input("Fasting Blood Sugar (>120 mg/dl)", "fbs", "toggle"),
        user_input("Resting ECG Results (0-2)", "restecg"),
        user_input("Max Heart Rate Achieved", "thalach"),
        user_input("Exercise Induced Angina", "exang", "toggle"),
        user_input("ST Depression", "oldpeak"),
        user_input("Slope of ST Segment", "slope"),
        user_input("Major Vessels Colored (0-4)", "ca"),
        user_input("Thalassemia (0-3)", "thal")
    ]
    if st.button("Check Heart Disease"):
        result = make_prediction(models['heart_disease'], features)
        st.success("Heart Disease Detected" if result == 1 else "No Heart Disease")

# Parkinson's Prediction
elif selected == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    features = [
        user_input(col, col) for col in [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 
            'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
    ]
    if st.button("Check Parkinson's"):
        result = make_prediction(models['parkinsons'], features)
        st.success("Parkinson's Detected" if result == 1 else "No Parkinson's")

# Lung Cancer Prediction
elif selected == "Lung Cancer":
    st.header("Lung Cancer Prediction")
    features = [
        user_input(col, col, "toggle") for col in [
            'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',
            'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
            'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
    ]
    if st.button("Check Lung Cancer"):
        result = make_prediction(models['lung_cancer'], features)
        st.success("Lung Cancer Detected" if result == 1 else "No Lung Cancer")

# Thyroid Prediction
if selected == "Thyroid":
    st.header("Thyroid Disease Prediction")
    features = [
        user_input("Age", "age"),
        user_input("Sex (1=Male, 0=Female)", "sex", "toggle"),
        user_input("On Thyroxine (1=Yes, 0=No)", "on_thyroxine", "toggle"),
        user_input("TSH Level", "TSH"),
        user_input("T3 Measured (1=Yes, 0=No)", "T3_measured", "toggle"),
        user_input("T3 Level", "T3"),
        user_input("TT4 Level", "TT4")
    ]
    if st.button("Check Thyroid"):
        result = make_prediction(models['thyroid'], features)
        st.success("Thyroid Disease Detected" if result == 1 else "No Thyroid Disease")
