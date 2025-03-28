import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
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

# Load datasets to determine feature names
data_files = {
    'diabetes': "diabetes_data.csv",
    'heart_disease': "heart_disease_data.csv",
    'parkinsons': "parkinson_data.csv",
    'lung_cancer': "survey_lung_cancer.csv",  # Fixed file name
    'thyroid': "hypothyroid.csv"
}

data_columns = {}
for disease, file in data_files.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        data_columns[disease] = df.columns[:-1].tolist()  # Exclude target column
    else:
        st.error(f"❌ Data file '{file}' not found!")
        st.stop()

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
def user_input(label, key):
    if "Yes" in label or "No" in label or "0/1" in label or "binary" in label.lower():
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

# Get the correct dataset key (lowercase, no spaces)
selected_key = selected.lower().replace(" ", "_")
if selected_key == "heart_disease":
    selected_key = "heart_disease"
elif selected_key == "lung_cancer":
    selected_key = "lung_cancer"
elif selected_key == "parkinsons":
    selected_key = "parkinsons"
elif selected_key == "thyroid":
    selected_key = "thyroid"

# Prediction logic based on selected disease
st.header(f"{selected} Prediction")

if selected_key in data_columns:
    features = [user_input(col, col) for col in data_columns[selected_key]]

    if st.button(f"Check {selected}"):
        result = make_prediction(models[selected_key], features)
        st.success(f"{selected} Detected" if result == 1 else f"No {selected}")
else:
    st.error("❌ Dataset not found for this prediction.")
