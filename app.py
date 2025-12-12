import streamlit as st
import json
import pandas as pd
import pickle
from io import StringIO
import random

from constants import ALL_COLUMNS

# constants
JSONS = ["Suhani_Bansal_Sample_1.json", "Suhani_Bansal_Sample_1.json", "Suhani_Bansal_Sample_1.json"]

IMAGE_ADDRESS = "https://biolabtests.com/wp-content/uploads/Microbial-Top-Facts-Klebsiella-pneumoniae.png"
# Add an image
st.image(IMAGE_ADDRESS, 
         caption="Classification")



st.set_page_config(page_title="K. pneumoniae • Ertapenem S/R Predictor", page_icon="🧬", layout="wide")

@st.cache_resource
def load_pickle_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# load models
model = load_pickle_model("LR")

st.title("Klebsiella pneumoniae – Ertapenem Susceptibility")
st.subheader("Predict Susceptible (S) or Resistant (R) from JSON features")
st.write(
        """
        This app loads a trained classifier and predicts Ertapenem susceptibility for
        Klebsiella pneumoniae. Upload a JSON with the required feature keys
        (e.g., spectrum_bin_*).
        Klebsiella pneumoniae is a type of Gram-negative, non-motile, rod-shaped bacterium that is part of the Enterobacteriaceae family. It is commonly found in the environment, including in soil, water, and plants, and can also be part of the normal flora in the human intestines. While it is harmless in the gut, it can cause a range of infections if it spreads to other parts of the body.Klebsiella pneumoniae is best known for causing pneumonia, particularly in hospital settings, where it is a significant cause of hospital-acquired infections. It can lead to symptoms such as fever, cough, chest pain, and difficulty breathing. 

        """
    )

# Sidebar for file upload
st.header("📤 Upload JSON Data")
uploaded_file = st.file_uploader(
    "Upload your spectral data (JSON only)",
    type=["json"],
    accept_multiple_files=False,
    help="Upload a JSON file containing spectral data"
)

with st.sidebar:
    st.subheader("Download Example Json")
    json_name = st.selectbox(
        "Select Example Json",
        JSONS,
    )
    with open(json_name, "r") as f:
        json_data = json.load(f)
    # disply the json if needed
    with st.expander("Example Json"):
        st.json(json_data)
    json_download_data = json.dumps(json_data, indent=4)
    st.download_button(
        label="Download Example Json",
        data=json_download_data,
        file_name=json_name,
        mime="application/json"
    )

# File processing logic
if uploaded_file is not None:
    try:
        # Read the uploaded file
        json_data = json.load(uploaded_file)
        
        # Display success message
        st.success("✅ File successfully uploaded and processed!")
        
        # Show a preview of the data
        with st.expander("📊 View Uploaded Data"):
            st.json(json_data)
            
        # Convert to DataFrame for better display (if the JSON structure is compatible)
        try:
            df = pd.json_normalize(json_data)
            columns_not_available = False
            for col in ALL_COLUMNS:
                if col not in df.columns:
                    columns_not_available = True
                    break
            if columns_not_available:
                st.error("❌ The uploaded JSON file does not contain all the required columns.", icon="⚠️")
                st.stop()
            
            # Add prediction button 
            if st.button("RUN", type="primary"):
                with st.spinner("Analyzing Spectral data..."):
                    # Placeholder for prediction logic
                    predictions = model.predict(df)
                    prediction_value= predictions[0]
                         
                    #st.header("Prediction: {}".format(predictions[0]))
                    # Map 0 → Resistant, 1 → Susceptible
                    if prediction_value == 0:
                     prediction_label = "Resistant (R)"
                    else:
                     prediction_label = "Susceptible (S)"
                    st.header(f"Prediction: {prediction_label}")     
                    
        except Exception as error:
            print(str(error))
            st.warning("Error in processing the file.", icon="⚠️")
            
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file. Please upload a valid JSON file.", icon="⚠️")
    except Exception as error:
        st.error(f"An error occurred: {str(error)}", icon="❌")
