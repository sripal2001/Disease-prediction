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
        st.error(f"\u274c Model file '{filename}' not found! Please check your repository.")
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
            st.error(f"\u274c Expected {model.n_features_in_} features, but got {len(features)}!")
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

# Thyroid Prediction
elif selected == "Thyroid":
    st.header("Thyroid Disease Prediction")
    features = [
        user_input("Age", "age"),
        user_input("Sex (1=Male, 0=Female)", "sex", "toggle"),
        user_input("On Thyroxine", "on thyroxine", "toggle"),
        user_input("Query on Thyroxine", "query on thyroxine", "toggle"),
        user_input("On Antithyroid Medication", "on antithyroid medication", "toggle"),
        user_input("Sick", "sick", "toggle"),
        user_input("Pregnant", "pregnant", "toggle"),
        user_input("Thyroid Surgery", "thyroid surgery", "toggle"),
        user_input("I131 Treatment", "I131 treatment", "toggle"),
        user_input("Query Hypothyroid", "query hypothyroid", "toggle"),
        user_input("Query Hyperthyroid", "query hyperthyroid", "toggle"),
        user_input("Lithium", "lithium", "toggle"),
        user_input("Goitre", "goitre", "toggle"),
        user_input("Tumor", "tumor", "toggle"),
        user_input("Hypopituitary", "hypopituitary", "toggle"),
        user_input("Psych", "psych", "toggle"),
        user_input("TSH Measured", "TSH measured", "toggle"),
        user_input("TSH", "TSH"),
        user_input("T3 Measured", "T3 measured", "toggle"),
        user_input("T3", "T3"),
        user_input("TT4 Measured", "TT4 measured", "toggle"),
        user_input("TT4", "TT4"),
        user_input("T4U Measured", "T4U measured", "toggle"),
        user_input("T4U", "T4U"),
        user_input("FTI Measured", "FTI measured", "toggle"),
        user_input("FTI", "FTI"),
        user_input("TBG Measured", "TBG measured", "toggle"),
        user_input("TBG", "TBG"),
    ]
    if st.button("Check Thyroid"):
        result = make_prediction(models['thyroid'], features)
        st.success("Thyroid Disease Detected" if result == 1 else "No Thyroid Disease")