import streamlit as st
import pickle
import numpy as np
import os
from streamlit_option_menu import option_menu

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

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Diabetes", "Heart Disease", "Parkinson's", "Lung Cancer", "Thyroid"],
        icons=["activity", "heart", "person", "lungs", "droplet"],
        menu_icon="hospital",
        default_index=0,
    )

# Function for user inputs
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

# Disease Prediction Sections
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

elif selected == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    features = [
        user_input("MDVP:Fo", "MDVP_Fo"),
        user_input("MDVP:Fhi", "MDVP_Fhi"),
        user_input("MDVP:Flo", "MDVP_Flo"),
        user_input("MDVP:Jitter", "MDVP_Jitter"),
        user_input("MDVP:Shimmer", "MDVP_Shimmer"),
        user_input("NHR", "NHR"),
        user_input("HNR", "HNR")
    ]
    if st.button("Check Parkinson's"):
        result = make_prediction(models['parkinsons'], features)
        st.success("Parkinson's Detected" if result == 1 else "No Parkinson's")

elif selected == "Lung Cancer":
    st.header("Lung Cancer Prediction")
    features = [
        user_input("Age", "age"),
        user_input("Smoking", "smoking", "toggle"),
        user_input("Yellow Fingers", "yellow_fingers", "toggle"),
        user_input("Coughing", "coughing", "toggle"),
        user_input("Shortness of Breath", "shortness_breath", "toggle"),
        user_input("Fatigue", "fatigue", "toggle"),
    ]
    if st.button("Check Lung Cancer"):
        result = make_prediction(models['lung_cancer'], features)
        st.success("Lung Cancer Detected" if result == 1 else "No Lung Cancer")
