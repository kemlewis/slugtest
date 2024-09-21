# Streamlit App Code

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
import joblib

# Load trained model, scaler, and label encoder
@st.cache(allow_output_mutation=True)
def load_models():
    clf = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    return clf, scaler, le

clf, scaler, le = load_models()

# Define hydraulic equations
def hvorslev(t, rc, Le, Re, K):
    T0 = (rc**2 * np.pi * Le) / (K * np.pi * Re**2)
    return np.exp(-t / T0)

def bouwer_rice(t, rc, Le, Re, K):
    ln_term = np.log(Re / rc)
    T0 = (Le * ln_term) / K
    return np.exp(-t / T0)

def cooper_bredehoeft(t, S, K, rc, Re):
    u = (rc**2 * S) / (4 * K * t)
    return erfc(np.sqrt(u))

def butler_zhan(t, K, S, rc, rw, H):
    alpha = (K * t) / (S * rc**2)
    return erfc(np.sqrt(alpha))

def van_der_kamp(t, K, S, rc, rw):
    alpha = (K * t) / (S * rc**2)
    return np.exp(-alpha)

equations = {
    'Bouwer-Rice': {
        'func': bouwer_rice,
        'params': ['rc', 'Le', 'Re', 'K']
    },
    'Butler-Zhan': {
        'func': butler_zhan,
        'params': ['K', 'S', 'rc', 'rw', 'H']
    },
    'Cooper-Bredehoeft': {
        'func': cooper_bredehoeft,
        'params': ['S', 'K', 'rc', 'Re']
    },
    'Hvorslev': {
        'func': hvorslev,
        'params': ['rc', 'Le', 'Re', 'K']
    },
    'van der Kamp': {
        'func': van_der_kamp,
        'params': ['K', 'S', 'rc', 'rw']
    }
}

# App Title
st.title("Slug Test Analysis App")

# Sidebar for user input
st.sidebar.header("Upload Slug Test Data")
uploaded_file = st.sidebar.file_uploader("Upload your slug test data (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Ensure required columns are present
    if 'Time' in data.columns and 'Response' in data.columns:
        t = data['Time'].values
        s = data['Response'].values

        st.write("### Uploaded Slug Test Data")
        st.line_chart(data.set_index('Time'))

        # Sidebar for additional parameters
        st.sidebar.header("Input Parameters")

        # User inputs for borehole details and parameter limits
        rc = st.sidebar.number_input('Casing Radius (rc) [m]', min_value=0.01, max_value=1.0, value=0.1)
        rw = st.sidebar.number_input('Well Radius (rw) [m]', min_value=0.01, max_value=1.0, value=0.1)
        Le = st.sidebar.number_input('Screen Length (Le) [m]', min_value=0.1, max_value=50.0, value=5.0)
        Re = st.sidebar.number_input('Effective Radius (Re) [m]', min_value=1.0, max_value=100.0, value=30.0)
        H = st.sidebar.number_input('Aquifer Thickness (H) [m]', min_value=1.0, max_value=100.0, value=20.0)

        # Parameter limits
        K_min = st.sidebar.number_input('Minimum Hydraulic Conductivity (K_min) [m/s]', min_value=1e-9, max_value=1e-2, value=1e-6, format="%.1e")
        K_max = st.sidebar.number_input('Maximum Hydraulic Conductivity (K_max) [m/s]', min_value=1e-9, max_value=1e-2, value=1e-3, format="%.1e")
        S_min = st.sidebar.number_input('Minimum Storage Coefficient (S_min)', min_value=1e-6, max_value=1e-1, value=1e-5, format="%.1e")
        S_max = st.sidebar.number_input('Maximum Storage Coefficient (S_max)', min_value=1e-6, max_value=1e-1, value=1e-2, format="%.1e")

        # Feature extraction
        def extract_features(t, s):
            features = []
            # Statistical features
            features.append(np.max(s))
            features.append(np.min(s))
            features.append(np.mean(s))
            features.append(np.std(s))
            features.append(np.trapz(s, t))  # Area under the curve
            features.append(np.percentile(s, 25))
            features.append(np.percentile(s, 50))
            features.append(np.percentile(s, 75))
            # Dynamic features
            derivative = np.gradient(s, t)
            features.append(np.max(derivative))
            features.append(np.min(derivative))
            features.append(np.mean(derivative))
            features.append(np.std(derivative))
            # Frequency domain features
            fft_vals = np.abs(np.fft.fft(s))
            features.append(np.max(fft_vals))
            features.append(np.mean(fft_vals))
            features = np.array(features).reshape(1, -1)
            return features

        # Extract features from user's data
        features = extract_features(t, s)
        X_scaled = scaler.transform(features)

        # Predict best-fitting equation
        probabilities = clf.predict_proba(X_scaled)[0]
        predicted_class = le.classes_[np.argmax(probabilities)]
        st.write(f"### Predicted Best-Fit Equation: **{predicted_class}**")

        # Show probabilities
        prob_df = pd.DataFrame({'Equation': le.classes_, 'Probability': probabilities})
        prob_df = prob_df.sort_values(by='Probability', ascending=False)
        st.write("### Equation Prediction Probabilities")
        st.table(prob_df)

        # Fit the predicted equation
        st.write("### Fitting Equations to Your Data")

        # Option to fit all equations or just the top N
        fit_all = st.checkbox("Fit All Equations", value=True)
        if fit_all:
            equations_to_fit = le.classes_
        else:
            N = st.number_input("Number of Top Equations to Fit", min_value=1, max_value=len(le.classes_), value=3)
            equations_to_fit = prob_df['Equation'].head(int(N)).values

        fit_results = {}
        for eq_name in equations_to_fit:
            eq_info = equations[eq_name]
            func = eq_info['func']
            params = eq_info['params']

            # Set up initial guesses and bounds
            initial_guesses = []
            bounds = ([], [])
            for param in params:
                if param == 'rc':
                    initial_guesses.append(rc)
                    bounds[0].append(rc * 0.9)
                    bounds[1].append(rc * 1.1)
                elif param == 'rw':
                    initial_guesses.append(rw)
                    bounds[0].append(rw * 0.9)
                    bounds[1].append(rw * 1.1)
                elif param == 'Le':
                    initial_guesses.append(Le)
                    bounds[0].append(Le * 0.9)
                    bounds[1].append(Le * 1.1)
                elif param == 'Re':
                    initial_guesses.append(Re)
                    bounds[0].append(Re * 0.9)
                    bounds[1].append(Re * 1.1)
                elif param == 'H':
                    initial_guesses.append(H)
                    bounds[0].append(H * 0.9)
                    bounds[1].append(H * 1.1)
                elif param == 'K':
                    initial_guesses.append((K_min + K_max) / 2)
                    bounds[0].append(K_min)
                    bounds[1].append(K_max)
                elif param == 'S':
                    initial_guesses.append((S_min + S_max) / 2)
                    bounds[0].append(S_min)
                    bounds[1].append(S_max)
                else:
                    initial_guesses.append(1.0)
                    bounds[0].append(0)
                    bounds[1].append(np.inf)

            # Fit the equation to the data
            try:
                popt, pcov = curve_fit(
                    func, t, s, p0=initial_guesses, bounds=bounds, maxfev=10000
                )
                s_fit = func(t, *popt)
                residuals = s - s_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((s - np.mean(s))**2)
                r_squared = 1 - (ss_res / ss_tot)
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))

                fit_results[eq_name] = {
                    'Parameters': dict(zip(params, popt)),
                    'R2': r_squared,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Fitted Curve': s_fit
                }
            except Exception as e:
                st.write(f"Error fitting {eq_name}: {e}")
                continue

        # Display fit results
        if fit_results:
            fit_df = pd.DataFrame.from_dict(fit_results, orient='index')
            st.write("### Fit Results")
            st.dataframe(fit_df[['R2', 'RMSE', 'MAE']])

            # Display fitted parameters
            st.write("### Fitted Parameters")
            for eq_name, result in fit_results.items():
                st.write(f"**{eq_name}:**")
                st.json(result['Parameters'])

            # Plot observed data and fitted curves
            st.write("### Observed Data and Fitted Curves")
            plt.figure(figsize=(10,6))
            plt.plot(t, s, label='Observed Data', color='black', linewidth=2)
            for eq_name, result in fit_results.items():
                plt.plot(t, result['Fitted Curve'], label=f"{eq_name} (R²={result['R2']:.2f})")
            plt.xlabel('Time')
            plt.ylabel('Response')
            plt.title('Slug Test Data and Fitted Curves')
            plt.legend()
            st.pyplot(plt)

        else:
            st.write("No equations were successfully fitted to your data.")

    else:
        st.error("Uploaded CSV must contain 'Time' and 'Response' columns.")
else:
    st.info("Please upload your slug test data to proceed.")
