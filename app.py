# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.special import erfc

# -----------------------------
# Section 1: Load Saved Components
# -----------------------------

@st.cache_resource
def load_model():
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    model = joblib.load('xgb_model.joblib')
    return scaler, label_encoder, model

scaler, label_encoder, model = load_model()
st.sidebar.success("Model components loaded successfully.")

# -----------------------------
# Section 2: Define Hydraulic Equations and Feature Extraction
# -----------------------------

def hvorslev(t, rc, Le, Re, K):
    T0 = (rc**2 * np.pi * Le) / (K * np.pi * Re**2)
    s = np.exp(-t / T0)
    return s

def bouwer_rice(t, rc, Le, Re, K):
    ln_term = np.log(Re / rc)
    T0 = (Le * ln_term) / K
    s = np.exp(-t / T0)
    return s

def cooper_bredehoeft(t, S, K, rc, Re):
    u = (rc**2 * S) / (4 * K * t)
    s = erfc(np.sqrt(u))
    return s

def butler_zhan(t, K, S, rc, rw, H):
    alpha = (K * t) / (S * rc**2)
    s = erfc(np.sqrt(alpha))
    return s

def van_der_kamp(t, K, S, rc, rw):
    alpha = (K * t) / (S * rc**2)
    s = np.exp(-alpha)
    return s

def generate_synthetic_data(equation, params, t):
    if equation == 'Hvorslev':
        s = hvorslev(t, **params)
    elif equation == 'Bouwer-Rice':
        s = bouwer_rice(t, **params)
    elif equation == 'Cooper-Bredehoeft':
        s = cooper_bredehoeft(t, **params)
    elif equation == 'Butler-Zhan':
        s = butler_zhan(t, **params)
    elif equation == 'van der Kamp':
        s = van_der_kamp(t, **params)
    else:
        raise ValueError("Unknown equation")
    return s

def extract_features(t, s):
    """
    Extract advanced features from the hydrograph data.
    """
    features = []
    for curve in s:
        feat = []
        # Statistical features
        feat.append(np.max(curve))
        feat.append(np.min(curve))
        feat.append(np.mean(curve))
        feat.append(np.std(curve))
        feat.append(np.trapz(curve, t))  # Area under the curve
        feat.append(np.percentile(curve, 25))
        feat.append(np.percentile(curve, 50))
        feat.append(np.percentile(curve, 75))
        # Dynamic features
        derivative = np.gradient(curve, t)
        feat.append(np.max(derivative))
        feat.append(np.min(derivative))
        feat.append(np.mean(derivative))
        feat.append(np.std(derivative))
        # Frequency domain features
        fft_vals = np.abs(np.fft.fft(curve))
        feat.append(np.max(fft_vals))
        feat.append(np.mean(fft_vals))
        features.append(feat)
    features = np.array(features)
    return features

# -----------------------------
# Section 3: Prediction Function
# -----------------------------

def predict_new_data(new_curves, t):
    """
    Predict the hydraulic equation for new synthetic curves.
    
    Parameters:
    - new_curves: list or array of 1D numpy arrays representing the synthetic curves.
    - t: 1D numpy array of time points corresponding to the curves.
    
    Returns:
    - predictions: List of predicted equation names.
    """
    # Ensure new_curves is a 2D array
    if isinstance(new_curves, list):
        new_curves = np.array(new_curves)
    elif isinstance(new_curves, np.ndarray) and new_curves.ndim == 1:
        new_curves = new_curves.reshape(1, -1)
    
    # Extract features
    features = extract_features(t, new_curves)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    y_pred_encoded = model.predict(features_scaled)
    
    # Decode labels
    predictions = label_encoder.inverse_transform(y_pred_encoded)
    
    return predictions

# -----------------------------
# Section 4: Streamlit UI
# -----------------------------

st.title("Hydraulic Equation Predictor")
st.write("""
This application predicts the hydraulic equation based on user-uploaded synthetic curve data.
""")

st.sidebar.header("Upload Your Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV file with synthetic curves", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.write(data.head())
        
        # Ensure data is in the correct format
        # Assuming each row is a separate curve and columns correspond to time points
        new_curves = data.values  # Shape: (num_samples, num_features)
        
        # Define the time vector (must match the one used during training)
        t_new = np.linspace(0.1, 10, 200)  # Adjust if different
        
        # Make predictions
        predictions = predict_new_data(new_curves, t_new)
        
        # Display predictions
        st.write("### Predictions")
        prediction_df = pd.DataFrame({
            'Curve Index': np.arange(1, len(predictions)+1),
            'Predicted Equation': predictions
        })
        st.write(prediction_df)
        
        # Optionally, plot each curve with its prediction
        st.write("### Curve Plots with Predictions")
        for i, curve in enumerate(new_curves):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t_new, curve, label='Uploaded Curve')
            ax.set_xlabel('Time')
            ax.set_ylabel('Response')
            ax.set_title(f'Curve {i+1}: Predicted Equation - {predictions[i]}')
            ax.legend()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

