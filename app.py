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

if selected == "Parkinson's":
    st.header("Parkinson's Disease Prediction")
    features = [
        user_input("MDVP:Fo (Fundamental Frequency)", "MDVP_Fo"),
        user_input("MDVP:Fhi (Highest Frequency)", "MDVP_Fhi"),
        user_input("MDVP:Flo (Lowest Frequency)", "MDVP_Flo"),
        user_input("MDVP:Jitter (%)", "MDVP_Jitter"),
        user_input("MDVP:Jitter (Abs)", "MDVP_Jitter_Abs"),
        user_input("MDVP:RAP (Relative Amplitude Perturbation)", "MDVP_RAP"),
        user_input("MDVP:PPQ (Pitch Period Perturbation Quotient)", "MDVP_PPQ"),
        user_input("Jitter:DDP", "Jitter_DDP"),
        user_input("MDVP:Shimmer", "MDVP_Shimmer"),
        user_input("MDVP:Shimmer (dB)", "MDVP_Shimmer_dB"),
        user_input("Shimmer:APQ3", "Shimmer_APQ3"),
        user_input("Shimmer:APQ5", "Shimmer_APQ5"),
        user_input("MDVP:APQ", "MDVP_APQ"),
        user_input("Shimmer:DDA", "Shimmer_DDA"),
        user_input("NHR (Noise-to-Harmonics Ratio)", "NHR"),
        user_input("HNR (Harmonics-to-Noise Ratio)", "HNR"),
    ]
    if st.button("Check for Parkinson's Disease"):
        result = make_prediction(models['parkinsons'], features)
        if result is not None:
            st.success("Parkinson's Detected" if result == 1 else "No Parkinson's")

elif selected == "Lung Cancer":
    st.header("Lung Cancer Prediction")
    features = [
        user_input("Age", "age"),
        user_input("Gender (1=Male, 0=Female)", "gender", "toggle"),
        user_input("Air Pollution Exposure", "air_pollution"),
        user_input("Alcohol Use", "alcohol_use"),
        user_input("Dust Allergy", "dust_allergy"),
        user_input("Occasional Coughing", "coughing"),
        user_input("Smoking History", "smoking"),
        user_input("Passive Smoking Exposure", "passive_smoking"),
        user_input("Genetic Risk", "genetic_risk"),
        user_input("Shortness of Breath", "shortness_breath"),
        user_input("Frequent Chest Pain", "chest_pain"),
    ]
    if st.button("Check for Lung Cancer"):
        result = make_prediction(models['lung_cancer'], features)
        if result is not None:
            st.success("Lung Cancer Detected" if result == 1 else "No Lung Cancer")

elif selected == "Thyroid":
    st.header("Thyroid Disease Prediction")
    features = [
        user_input("Age", "age"),
        user_input("Gender (1=Male, 0=Female)", "gender", "toggle"),
        user_input("TSH (Thyroid-Stimulating Hormone)", "TSH"),
        user_input("T3 (Triiodothyronine)", "T3"),
        user_input("TT4 (Total Thyroxine)", "TT4"),
        user_input("T4U (Thyroxine Utilization)", "T4U"),
        user_input("FTI (Free Thyroxine Index)", "FTI"),
        user_input("On Thyroxine Medication", "on_thyroxine", "toggle"),
        user_input("Query Hyperthyroid", "query_hyperthyroid", "toggle"),
        user_input("Query Hypothyroid", "query_hypothyroid", "toggle"),
    ]
    if st.button("Check for Thyroid Disease"):
        result = make_prediction(models['thyroid'], features)
        if result is not None:
            st.success("Thyroid Disease Detected" if result == 1 else "No Thyroid Disease")
