import streamlit as st
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from statistics import mode

from constants import ALL_COLUMNS

# constants
JSONS = ["Sample_1.json", "Sample_2.json", "Sample_3.json","Sample_4.json","Sample_5.json"]

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

# load all three models
logistic_regression_model = load_pickle_model("LR")
mlp_model = load_pickle_model("MLP")
rf_model = load_pickle_model("RandomForest")

# Predefined model performance metrics (you should replace these with actual metrics from your models)
MODEL_METRICS = {
    'Logistic Regression': {
        'precision': 0.85,
        'recall': 0.82,
        'f1_score': 0.83,
        'accuracy': 0.88
    },
    'Multi-Layer Perceptron': {
        'precision': 0.87,
        'recall': 0.84,
        'f1_score': 0.85,
        'accuracy': 0.89
    },
    'K-Nearest Neighbors': {
        'precision': 0.83,
        'recall': 0.80,
        'f1_score': 0.81,
        'accuracy': 0.86
    }
}

st.title("Klebsiella pneumoniae – Ertapenem Susceptibility")
st.subheader("Predict Susceptible (S) or Resistant (R) from JSON features")
st.write(
    """
    This app loads three trained classifiers and predicts Ertapenem susceptibility for
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
    # display the json if needed
    with st.expander("Example Json"):
        st.json(json_data)
    json_download_data = json.dumps(json_data, indent=4)
    st.download_button(
        label="Download Example Json",
        data=json_download_data,
        file_name=json_name,
        mime="application/json"
    )

# Function to get prediction label
def get_prediction_label(prediction_value):
    """Convert numeric prediction to readable label"""
    if prediction_value == 0:
        return "Resistant"
    elif prediction_value == 1:
        return "Susceptible (S)"
    else:
        return f"Unknown ({prediction_value})"

# Function to get short prediction label for table
def get_short_prediction_label(prediction_value):
    """Get short prediction label for table"""
    if prediction_value == 0:
        return "Resistant"
    elif prediction_value == 1:
        return "S"
    else:
        return "Unknown"

# Function to get prediction confidence/probability
def get_prediction_confidence(model, df, prediction_value):
    """Get prediction probability if model supports it"""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
            confidence = probabilities[0][prediction_value] * 100
            return confidence
    except:
        pass
    return None

# File processing logic
if uploaded_file is not None:
    try:
        # Read the uploaded file
        json_data = json.load(uploaded_file)
        
        # Get sample ID from filename
        sample_id = uploaded_file.name.replace('.json', '')
        
        # Display success message
        st.success(f"✅ File '{sample_id}' successfully uploaded and processed!")
        
        # Show a preview of the data
        with st.expander("📊 View Uploaded Data"):
            st.json(json_data)
            
        # Convert to DataFrame for better display
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
            if st.button("RUN PREDICTION", type="primary"):
                with st.spinner("Analyzing Spectral data with all models..."):
                    # Get predictions from all three models
                    lr_prediction = logistic_regression_model.predict(df)[0]
                    mlp_prediction = mlp_model.predict(df)[0]
                    rf_prediction = rf_model.predict(df)[0]
                    
                    # Get confidence scores
                    lr_confidence = get_prediction_confidence(logistic_regression_model, df, lr_prediction)
                    mlp_confidence = get_prediction_confidence(mlp_model, df, mlp_prediction)
                    rf_confidence = get_prediction_confidence(rf_model, df, rf_prediction)
                    
                    # Create results table matching the image format
                    st.header("📊 Prediction Results Table")
                    
                    # Create a DataFrame for the results table
                    results_data = []
                    
                    # Add Logistic Regression results
                    results_data.append({
                        'Sample ID': sample_id,
                        'Model': 'Logical Regression',  # Matching your image spelling
                        'Precision': f"{MODEL_METRICS['Logistic Regression']['precision']:.1f}",
                        'Recall': f"{MODEL_METRICS['Logistic Regression']['recall']:.1f}",
                        'F1-Score': f"{MODEL_METRICS['Logistic Regression']['f1_score']:.1f}",
                        'Accuracy': f"{MODEL_METRICS['Logistic Regression']['accuracy']:.1f}",
                        'Final Outcome': get_short_prediction_label(lr_prediction)
                    })
                    
                    # Add MLP results
                    results_data.append({
                        'Sample ID': sample_id,
                        'Model': 'MLP',
                        'Precision': f"{MODEL_METRICS['Multi-Layer Perceptron']['precision']:.1f}",
                        'Recall': f"{MODEL_METRICS['Multi-Layer Perceptron']['recall']:.1f}",
                        'F1-Score': f"{MODEL_METRICS['Multi-Layer Perceptron']['f1_score']:.1f}",
                        'Accuracy': f"{MODEL_METRICS['Multi-Layer Perceptron']['accuracy']:.1f}",
                        'Final Outcome': get_short_prediction_label(mlp_prediction)
                    })
                    
                    # Add KNN results (using "RF" from your image if needed, but showing KNN)
                    results_data.append({
                        'Sample ID': sample_id,
                        'Model': 'RF',  # Using RF as shown in image, change to KNN if needed
                        'Precision': f"{MODEL_METRICS['K-Nearest Neighbors']['precision']:.1f}",
                        'Recall': f"{MODEL_METRICS['K-Nearest Neighbors']['recall']:.1f}",
                        'F1-Score': f"{MODEL_METRICS['K-Nearest Neighbors']['f1_score']:.1f}",
                        'Accuracy': f"{MODEL_METRICS['K-Nearest Neighbors']['accuracy']:.1f}",
                        'Final Outcome': get_short_prediction_label(rf_prediction)
                    })
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Display the table with professional formatting
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Sample ID": st.column_config.TextColumn(width="small"),
                            "Model": st.column_config.TextColumn(width="medium"),
                            "Precision": st.column_config.NumberColumn(format="%.1f"),
                            "Recall": st.column_config.NumberColumn(format="%.1f"),
                            "F1-Score": st.column_config.NumberColumn(format="%.1f"),
                            "Accuracy": st.column_config.NumberColumn(format="%.1f"),
                            "Final Outcome": st.column_config.TextColumn(width="small")
                        }
                    )
                    
                    # Add download button for results
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"prediction_results_{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Determine final consensus
                    st.divider()
                    st.subheader("🎯 Consensus Analysis")
                    
                    predictions = [lr_prediction, mlp_prediction, rf_prediction]
                    final_prediction_value=mode(predictions)
                    final_prediction=get_prediction_label(final_prediction_value)
                    prediction_icon = "✅" if final_prediction_value == 1 else "❌"
                    prediction_color = "green" if final_prediction_value == 1 else "red"
                  
                    # Display consensus result
                    col1,= st.columns(1)
                    with col1:
                        st.markdown(f"<h2 style='color:{prediction_color}; text-align: center;'>{prediction_icon}</h2>", 
                                  unsafe_allow_html=True)
                        st.markdown(f"<h3 style='color:{prediction_color}; text-align: center;'>{final_prediction}</h3>", 
                                  unsafe_allow_html=True)
                    
                    
                    # Add explanation and disclaimer
                    st.divider()
                    st.subheader("📋 Clinical Notes")
                    explanation_text = f"""
                    
                    ### ⚠️ Important Clinical Disclaimer
                    
                    **This tool is for research and informational purposes only.**
                     
                    1. **Not a Diagnostic Tool:** This prediction should not be used as the sole determinant for clinical decision-making.
                    2. **Clinical Judgment:** Treatment decisions must be made by qualified healthcare professionals considering patient history, comorbidities, and local resistance patterns.
                    3. **Regulatory Status:** This software is intended for research use only.
                                       
                    Always consult with infectious disease specialists for treatment decisions involving antimicrobial therapy.
                    """
                    
                    st.markdown(explanation_text)
                    
        except Exception as error:
            print(str(error))
            st.warning("Error in processing the file.", icon="⚠️")
            
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file. Please upload a valid JSON file.", icon="⚠️")
    except Exception as error:
        st.error(f"An error occurred: {str(error)}", icon="❌")
