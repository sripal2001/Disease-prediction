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
    'diabetes': "/mnt/data/diabetes_data.csv",
    'heart_disease': "/mnt/data/heart_disease_data.csv",
    'parkinsons': "/mnt/data/parkinson_data.csv",
    'lung_cancer': "/mnt/data/survey lung cancer.csv",
    'thyroid': "/mnt/data/hypothyroid.csv"
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

# Prediction logic based on selected disease
st.header(f"{selected} Prediction")

features = [
    user_input(col, col, "toggle" if "Yes/No" in col or "0/1" in col else "number")
    for col in data_columns[selected.lower().replace(" ", "_")]
]

if st.button(f"Check {selected}"):
    result = make_prediction(models[selected.lower().replace(" ", "_")], features)
    st.success(f"{selected} Detected" if result == 1 else f"No {selected}")