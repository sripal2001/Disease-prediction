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
    features = [
        user_input("Number of Pregnancies", "Pregnancies"),
        user_input("Glucose Level", "Glucose"),
        user_input("Blood Pressure", "BloodPressure"),
        user_input("Skin Thickness", "SkinThickness"),
        user_input("Insulin Level", "Insulin"),
        user_input("BMI", "BMI", "slider"),
        user_input("Diabetes Pedigree Function", "DiabetesPedigreeFunction"),
        user_input("Age", "Age", "slider")
    ]
    if st.button("Check Diabetes"):
        result = make_prediction(models['diabetes'], features)
        st.success("Diabetic" if result == 1 else "Not Diabetic")

elif selected == "Heart Disease":
    st.header("Heart Disease Prediction")
    features = [
        user_input("Age", "age", "slider"),
        user_input("Sex (1=Male, 0=Female)", "sex", "toggle"),
        user_input("Chest Pain Type (0-3)", "cp"),
        user_input("Resting Blood Pressure", "trestbps"),
        user_input("Serum Cholesterol (mg/dl)", "chol"),
        user_input("Fasting Blood Sugar > 120 mg/dl", "fbs", "toggle"),
        user_input("Resting ECG Results (0-2)", "restecg"),
        user_input("Max Heart Rate Achieved", "thalach"),
        user_input("Exercise Induced Angina", "exang", "toggle"),
        user_input("ST Depression Induced by Exercise", "oldpeak"),
        user_input("Slope of Peak Exercise ST Segment (0-2)", "slope"),
        user_input("Number of Major Vessels Colored by Fluoroscopy (0-4)", "ca"),
        user_input("Thalassemia (0-3)", "thal")
    ]
    if st.button("Check Heart Disease"):
        result = make_prediction(models['heart_disease'], features)
        st.success("Heart Disease Detected" if result == 1 else "No Heart Disease")

elif selected == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    features = [user_input(f"Feature {i+1}", f"parkinson_{i+1}") for i in range(22)]
    if st.button("Check Parkinson's"):
        result = make_prediction(models['parkinsons'], features)
        st.success("Parkinson's Detected" if result == 1 else "No Parkinson's")

elif selected == "Lung Cancer":
    st.header("Lung Cancer Prediction")
    features = [user_input(f"Feature {i+1}", f"lung_cancer_{i+1}") for i in range(10)]
    if st.button("Check Lung Cancer"):
        result = make_prediction(models['lung_cancer'], features)
        st.success("Lung Cancer Detected" if result == 1 else "No Lung Cancer")

elif selected == "Thyroid":
    st.header("Thyroid Disease Prediction")
    features = [user_input(f"Feature {i+1}", f"thyroid_{i+1}") for i in range(7)]
    if st.button("Check Thyroid"):
        result = make_prediction(models['thyroid'], features)
        st.success("Thyroid Issue Detected" if result == 1 else "No Thyroid Issue")

st.markdown("### Stay Healthy 💖")
