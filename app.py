import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
import os

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

def user_input(label, key, type="number"):
    if type == "toggle":
        return int(st.toggle(label, key=key))
    elif type == "slider":
        return int(st.slider(label, 0, 100, step=1, key=key))
    return float(st.number_input(label, step=1.0, key=key))

def make_prediction(model, features):
    try:
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

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
        result = make_prediction(models['diabetes'], [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if result is not None:
            st.success("Diabetic" if result == 1 else "Not Diabetic")

elif selected == "Heart Disease":
    st.header("Heart Disease Prediction")
    col1, col2 = st.columns(2)
    with col1:
        age = user_input("Age", "age", "slider")
        sex = user_input("Sex (1=Male, 0=Female)", "sex", "toggle")
        cp = user_input("Chest Pain Type (0-3)", "cp")
        trestbps = user_input("Resting Blood Pressure", "trestbps")
        restecg = user_input("Resting ECG Results (0-2)", "restecg")
    with col2:
        chol = user_input("Serum Cholesterol (mg/dl)", "chol")
        fbs = user_input("Fasting Blood Sugar > 120 mg/dl", "fbs", "toggle")
        thalach = user_input("Max Heart Rate Achieved", "thalach")
        exang = user_input("Exercise Induced Angina", "exang", "toggle")
        oldpeak = user_input("ST Depression Induced by Exercise", "oldpeak")
        slope = user_input("Slope of Peak Exercise ST Segment (0-2)", "slope")
        ca = user_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", "ca")
        thal = user_input("Thalassemia (0-3)", "thal")
    
    if st.button("Check Heart Disease"):
        result = make_prediction(models['heart_disease'], [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        if result is not None:
            st.success("Heart Disease Detected" if result == 1 else "No Heart Disease")

st.markdown("### Stay Healthy 💖")
