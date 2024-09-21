import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import curve_fit
from scipy.special import erfc
import joblib
from functools import partial

# Load trained model, scaler, and label encoder
@st.cache(allow_output_mutation=True)
def load_models():
    clf = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    return clf, scaler, le

clf, scaler, le = load_models()

# Define hydraulic equations with dynamic parameters
def hvorslev(t, K, rc, Le, Re):
    """
    Hvorslev Equation
    """
    return 1 - np.exp(-((2 * K * t) / (rc**2 * np.log(Le / rc))))

def bouwer_rice(t, K, rc, Le, Re):
    """
    Bouwer-Rice Equation
    """
    return 1 - np.exp(-((2 * K * t) / (rc**2 * np.log(Re / rc))))

def cooper_bredehoeft(t, T, S, rc):
    """
    Cooper-Bredehoeft Equation
    """
    return 1 - erfc(np.sqrt((4 * T * t) / (S * rc**2)))

def butler_zhan(t, T, S, rc):
    """
    Butler-Zhan Equation
    """
    return 1 - erfc(np.sqrt((4 * T * t) / (S * rc**2)))

def van_der_kamp(t, T, S, rc):
    """
    van der Kamp Equation
    """
    return 1 - np.exp(-(4 * T * t) / (S * rc**2))

# Map equation names to functions and their fitting parameters
equations = {
    'Bouwer-Rice': {'func': bouwer_rice, 'params': ['K']},
    'Butler-Zhan': {'func': butler_zhan, 'params': ['T', 'S']},
    'Cooper-Bredehoeft': {'func': cooper_bredehoeft, 'params': ['T', 'S']},
    'Hvorslev': {'func': hvorslev, 'params': ['K']},
    'van der Kamp': {'func': van_der_kamp, 'params': ['T', 'S']}
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

        # Check if time is in seconds, if not, convert
        if np.max(t) < 1:  # Assuming time is in minutes if max is less than 1
            t = t * 60  # Convert to seconds

        # Normalize the response data
        s_norm = (s - np.min(s)) / (np.max(s) - np.min(s))

        st.write("### Uploaded Slug Test Data")
        st.line_chart(data.set_index('Time'))

        # Sidebar for additional parameters
        st.sidebar.header("Input Parameters")

        # User inputs for borehole details and parameter limits
        rc = st.sidebar.number_input('Casing Radius (rc) [m]', min_value=0.01, max_value=1.0, value=0.1)
        Le = st.sidebar.number_input('Screen Length (Le) [m]', min_value=0.1, max_value=50.0, value=5.0)
        Re = st.sidebar.number_input('Effective Radius (Re) [m]', min_value=1.0, max_value=100.0, value=30.0)

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
        features = extract_features(t, s_norm)
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

            # Define fixed parameters using partial
            if eq_name in ['Bouwer-Rice', 'Hvorslev']:
                # These equations require rc, Le, Re
                func_fixed = partial(func, rc=rc, Le=Le, Re=Re)
            elif eq_name in ['Butler-Zhan', 'Cooper-Bredehoeft', 'van der Kamp']:
                # These equations require rc only
                func_fixed = partial(func, rc=rc)
            else:
                func_fixed = func  # Default case, should not occur

            # Set up initial guesses and bounds based on parameters
            if params == ['K']:
                initial_guesses = [1e-5]
                bounds_lower = [K_min]
                bounds_upper = [K_max]
            elif params == ['T', 'S']:
                initial_guesses = [1e-4, (S_min + S_max) / 2]
                bounds_lower = [1e-7, S_min]
                bounds_upper = [1e-1, S_max]
            else:
                st.write(f"Unknown parameters for {eq_name}. Skipping.")
                continue

            # Fit the equation to the normalized data
            try:
                popt, _ = curve_fit(
                    func_fixed, t, s_norm,
                    p0=initial_guesses,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=10000
                )
                s_fit = func_fixed(t, *popt)
                residuals = s_norm - s_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((s_norm - np.mean(s_norm))**2)
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
                st.warning(f"Error fitting {eq_name}: {e}")
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

            # Create interactive plot using Plotly Express
            st.write("### Observed Data and Fitted Curves")

            # Create a DataFrame for plotting
            plot_df = pd.DataFrame({'Time': t, 'Observed': s_norm})
            for eq_name, result in fit_results.items():
                plot_df[eq_name] = result['Fitted Curve']

            fig = px.line(plot_df, x='Time', y=[col for col in plot_df.columns if col != 'Time'],
                          title='Slug Test Data and Fitted Curves',
                          labels={'value': 'Normalized Response', 'Time': 'Time (seconds)'})

            # Update layout for better aesthetics
            fig.update_layout(
                legend_title="Data Series",
                xaxis_title="Time (seconds)",
                yaxis_title="Normalized Response",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.write("No equations were successfully fitted to your data.")
    else:
        st.error("Uploaded CSV must contain 'Time' and 'Response' columns.")
else:
    st.info("Please upload your slug test data to proceed.")
