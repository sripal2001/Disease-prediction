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
