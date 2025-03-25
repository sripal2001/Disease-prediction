import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import numpy as np

# Page Configurations
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Hide Streamlit branding
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import os

def load_model(filename):
    """Load a model file safely."""
    if not os.path.exists(filename):
        st.error(f"❌ Model file '{filename}' not found! Please check your GitHub repository.")
        st.stop()
    return pickle.load(open(filename, 'rb'))

models = {
    'diabetes': load_model("best_diabetes_model.sav"),
    'heart_disease': load_model("heart_disease_model.sav"),
    'parkinsons': load_model("parkinsons_model.sav"),
    'lung_cancer': load_model("lungs_disease_model.sav"),
    'thyroid': load_model("Thyroid_model.sav")
}
# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Diabetes", "Heart Disease", "Parkinson's", "Lung Cancer", "Thyroid"],
        icons=["activity", "heart", "person", "lungs", "droplet"],
        menu_icon="hospital",
        default_index=0,
    )

# Function to create UI Components
def user_input(label, key, type="number"):
    if type == "toggle":
        return st.toggle(label, key=key)
    elif type == "slider":
        return st.slider(label, 0, 100, step=1, key=key)
    return st.number_input(label, step=1, key=key)

# Prediction Logic
if selected == "Diabetes":
    st.header("Diabetes Prediction")
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = user_input("Number of Pregnancies", "Pregnancies")
        Glucose = user_input("Glucose Level", "Glucose")
        BloodPressure = user_input("Blood Pressure", "BloodPressure")
        SkinThickness = user_input("Skin Thickness", "SkinThickness")
    with col2:
        Insulin = user_input("Insulin Level", "Insulin")
        BMI = user_input("BMI", "BMI", "slider")
        DiabetesPedigreeFunction = user_input("Diabetes Pedigree Function", "DiabetesPedigreeFunction")
        Age = user_input("Age", "Age", "slider")
    
    if st.button("Check Diabetes"):
        prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(result)

elif selected == "Heart Disease":
    st.header("Heart Disease Prediction")
    col1, col2 = st.columns(2)
    with col1:
        age = user_input("Age", "age", "slider")
        sex = user_input("Sex (1=Male, 0=Female)", "sex", "toggle")
        cp = user_input("Chest Pain Type (0-3)", "cp")
        trestbps = user_input("Resting Blood Pressure", "trestbps")
    with col2:
        chol = user_input("Serum Cholesterol (mg/dl)", "chol")
        fbs = user_input("Fasting Blood Sugar > 120 mg/dl", "fbs", "toggle")
        thalach = user_input("Max Heart Rate Achieved", "thalach")
        exang = user_input("Exercise Induced Angina", "exang", "toggle")
    
    if st.button("Check Heart Disease"):
        prediction = models['heart_disease'].predict([[age, sex, cp, trestbps, chol, fbs, thalach, exang]])
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        st.success(result)

elif selected == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    fo = user_input("MDVP:Fo(Hz)", "fo")
    fhi = user_input("MDVP:Fhi(Hz)", "fhi")
    flo = user_input("MDVP:Flo(Hz)", "flo")
    jitter = user_input("MDVP:Jitter(%)", "jitter")
    if st.button("Check Parkinson's"):
        prediction = models['parkinsons'].predict([[fo, fhi, flo, jitter]])
        result = "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's"
        st.success(result)

elif selected == "Lung Cancer":
    st.header("Lung Cancer Prediction")
    smoking = user_input("Smoking", "smoking", "toggle")
    coughing = user_input("Coughing", "coughing", "toggle")
    chest_pain = user_input("Chest Pain", "chest_pain", "toggle")
    if st.button("Check Lung Cancer"):
        prediction = models['lung_cancer'].predict([[smoking, coughing, chest_pain]])
        result = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer"
        st.success(result)

elif selected == "Thyroid":
    st.header("Thyroid Disease Prediction")
    age = user_input("Age", "age", "slider")
    sex = user_input("Sex (1=Male, 0=Female)", "sex", "toggle")
    tsh = user_input("TSH Level", "tsh")
    if st.button("Check Thyroid"):
        prediction = models['thyroid'].predict([[age, sex, tsh]])
        result = "Thyroid Detected" if prediction[0] == 1 else "No Thyroid Disease"
        st.success(result)

st.markdown("### Stay Healthy 💖")
