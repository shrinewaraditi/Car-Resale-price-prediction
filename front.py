# app.py
import os
import re
import json
from datetime import datetime
import joblib
import requests

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Car Resale Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Styles ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .summary-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Session state ----------------
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

# ---------------- Helpers ----------------
def parse_torque(torque_str):
    """Extract numeric torque value from string like '113 Nm @ 4000 rpm'"""
    try:
        if torque_str is None:
            return 0.0
        match = re.search(r'(\d+\.?\d*)\s*(?:Nm|nm|NM)', str(torque_str))
        if match:
            return float(match.group(1))
        m2 = re.search(r'(\d+\.?\d*)', str(torque_str))
        return float(m2.group(1)) if m2 else 0.0
    except:
        return 0.0

def validate_inputs(data):
    """Validate inputs and return list of error strings (empty if ok)"""
    errors = []
    current_year = datetime.now().year
    if data['year'] > current_year:
        errors.append(f"⚠️ Year cannot be greater than {current_year}")
    if data['km_driven'] < 0:
        errors.append("⚠️ Kilometers driven cannot be negative")
    if data['engine'] < 50 or data['engine'] > 8000:
        errors.append("⚠️ Engine capacity should be between 50 and 8000 cc")
    if data['mileage'] <= 0:
        errors.append("⚠️ Mileage must be positive")
    if data['max_power'] <= 0:
        errors.append("⚠️ Max power must be positive")
    return errors

# ---------------- Cached model loader ----------------
@st.cache_resource
def load_model_and_info(model_path="car_price_model.pkl", info_path="model_info.pkl", refdata_path="reference_data.pkl"):
    model, model_info, ref_df = None, None, None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Failed to load model file '{model_path}': {e}")
    if os.path.exists(info_path):
        try:
            model_info = joblib.load(info_path)
        except Exception as e:
            st.error(f"Failed to load model info '{info_path}': {e}")
    if os.path.exists(refdata_path):
        try:
            ref_df = pd.read_pickle(refdata_path)
        except Exception:
            ref_df = None
    return model, model_info, ref_df

model, model_info, reference_df = load_model_and_info()

# quick sidebar notice about model
if model is None:
    st.sidebar.warning("Local model not found. Predictions will fallback to mock logic. Place 'car_price_model.pkl' and 'model_info.pkl' in the app folder for real inference.")
else:
    st.sidebar.success("Local model loaded ✔️")

# ---------------- Prediction functions ----------------
def mock_predict(data):
    """Fallback mock logic (keeps UI working if model not present)."""
    age = datetime.now().year - data['year']
    if age == 0: age = 0.5
    base_price = 500000
    age_factor = -age * 20000
    km_factor = -data['km_driven'] * 2
    fuel_adjustment = {
        'Petrol': 0, 'Diesel': 50000, 'CNG': -30000, 
        'LPG': -20000, 'Electric': 150000, 'Hybrid': 100000, 'Other': 0
    }
    trans_adjustment = {'Manual': 0, 'Automatic': 80000, 'Semi-Auto': 50000}
    owner_adjustment = {'First': 50000, 'Second': 0, 'Third': -30000,
                        'Fourth & Above': -50000, 'Test-Drive/Company': 30000}
    seller_adjustment = {'Individual': 0, 'Dealer': 20000, 'Trustmark/Certified Dealer': 40000}
    predicted = (base_price + age_factor + km_factor + 
                fuel_adjustment.get(data['fuel'], 0) +
                trans_adjustment.get(data['transmission'], 0) +
                owner_adjustment.get(data['owner'], 0) +
                seller_adjustment.get(data['seller_type'], 0) +
                data['engine'] * 50 + data['max_power'] * 1000)
    predicted = max(50000, predicted)
    lower = predicted * 0.85
    upper = predicted * 1.15
    confidence = 0.85
    explanation = {
        'top_features': [
            {'feature': 'Vehicle Age', 'impact': age_factor, 'direction': 'negative'},
            {'feature': 'Transmission Type', 'impact': trans_adjustment.get(data['transmission'], 0), 'direction': 'positive'},
            {'feature': 'Fuel Type', 'impact': fuel_adjustment.get(data['fuel'], 0),
             'direction': 'positive' if fuel_adjustment.get(data['fuel'], 0) > 0 else 'negative'}
        ]
    }
    return {
        'predicted_price': float(predicted),
        'lower': float(lower),
        'upper': float(upper),
        'confidence': float(confidence),
        'explanation': explanation
    }

def real_predict(input_data, model, model_info, reference_df, CURRENT_YR=None):
    """
    Preprocess input_data to match training features and call the loaded model.
    Returns same dict format as mock_predict.
    """
    if model is None or model_info is None:
        raise FileNotFoundError("Model or model_info not loaded")

    feature_cols = model_info.get('feature_columns', None)
    if feature_cols is None:
        raise KeyError("feature_columns missing in model_info.pkl")

    now_year = CURRENT_YR if CURRENT_YR is not None else datetime.now().year
    year = int(input_data.get('year', now_year))
    Age = now_year - year
    if Age == 0:
        Age = 0.5
    km_driven = float(input_data.get('km_driven', 0.0))
    KM_per_yr = km_driven / Age if Age > 0 else km_driven

    # transmission encoding used in training: Manual:0, Automatic:1
    trans_map = {'Manual': 0, 'Automatic': 1, 'Semi-Auto': 0}
    transmission_encoded = trans_map.get(input_data.get('transmission', 'Manual'), 0)

    fuel = input_data.get('fuel', 'Unknown')
    diesel = 1 if fuel == 'Diesel' else 0
    petrol = 1 if fuel == 'Petrol' else 0
    lpg = 1 if fuel == 'LPG' else 0
    cng = 1 if fuel == 'CNG' else 0

    owner = input_data.get('owner', 'First')
    o1 = 1 if owner.startswith('First') else 0
    o2 = 1 if owner.startswith('Second') else 0
    o3 = 1 if owner.startswith('Third') else 0
    o4 = 1 if 'Fourth' in owner or 'Fourth & Above' in owner else 0
    o_test = 1 if 'Test' in owner or 'Company' in owner else 0

    seller = input_data.get('seller_type', 'Individual')
    ind = 1 if seller == 'Individual' else 0
    dea = 1 if seller == 'Dealer' else 0
    trust = 1 if 'Trust' in seller else 0

    # numeric fields
    try:
        mileage_val = float(input_data.get('mileage', 0.0))
    except:
        mileage_val = 0.0
    try:
        engine_cc_val = int(input_data.get('engine', 0))
    except:
        engine_cc_val = 0
    try:
        max_power_val = float(input_data.get('max_power', 0.0))
    except:
        max_power_val = 0.0
    try:
        torque_val = float(input_data.get('torque', 0.0))
    except:
        torque_val = 0.0
    seats_no = int(input_data.get('seats', 0))

    feature_dict = {
        'Age': Age,
        'km_driven': km_driven,
        'KM_per_yr': KM_per_yr,
        'transmission': transmission_encoded,
        'diesel': diesel,
        'petrol': petrol,
        'lpg': lpg,
        'cng': cng,
        '1st own': o1,
        '2nd own': o2,
        '3rd own': o3,
        '4+ own': o4,
        'test drive own': o_test,
        'individual': ind,
        'dealer': dea,
        'trustmark dealer': trust,
        'Mileage_value': mileage_val,
        'Engine_cc': engine_cc_val,
        'max_power': max_power_val,
        'torque_value': torque_val,
        'seats': seats_no
    }

    X_user = pd.DataFrame([feature_dict])
    # ensure all expected columns exist in the right order
    for c in feature_cols:
        if c not in X_user.columns:
            X_user[c] = 0.0
    X_user = X_user[feature_cols].astype(float)

    try:
        raw_pred = float(model.predict(X_user)[0])
    except NotFittedError:
        raise RuntimeError("Loaded model is not fitted.")
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # Brand adjustment using reference_df if provided
    brand = (input_data.get('name') or "").split()[0].lower() if input_data.get('name') else ""
    predicted_price = raw_pred
    if reference_df is not None and brand:
        try:
            brands = reference_df['name'].dropna().str.lower().str.split().str[0].unique()
            if brand in brands:
                brand_median_price = reference_df[reference_df['name'].str.lower().str.startswith(brand)]['selling_price'].median()
                if not pd.isna(brand_median_price):
                    predicted_price = 0.5 * raw_pred + 0.5 * brand_median_price
        except Exception:
            pass

    lower = max(0.0, predicted_price * 0.85)
    upper = predicted_price * 1.15
    confidence = 0.85

    explanation = {
        'top_features': [
            {'feature': 'Vehicle Age', 'impact': -abs((datetime.now().year - year) * 20000), 'direction': 'negative'},
            {'feature': 'Engine (cc)', 'impact': engine_cc_val * 50, 'direction': 'positive'},
            {'feature': 'Transmission', 'impact': 80000 if transmission_encoded == 1 else 0, 'direction': 'positive' if transmission_encoded == 1 else 'negative'}
        ]
    }

    return {
        'predicted_price': float(predicted_price),
        'lower': float(lower),
        'upper': float(upper),
        'confidence': float(confidence),
        'explanation': explanation
    }

# ---------------- UI Layout ----------------
st.markdown("""
<div class="main-header">
    <h1>🚗 Car Resale Price Predictor</h1>
    <p>Professional AI-Powered Vehicle Valuation System</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    use_api = st.checkbox("Use API for prediction", value=False)
    api_endpoint = ""
    api_key = ""
    if use_api:
        api_endpoint = st.text_input("API Endpoint", placeholder="https://api.example.com/predict")
        api_key = st.text_input("API Key (optional)", type="password")
    include_asking_price = st.checkbox("Include asking price as input", value=False)
    st.subheader("📏 Unit Preferences")
    mileage_unit = st.selectbox("Mileage Unit", ["km/l", "km/kg"])
    power_unit = st.selectbox("Power Unit", ["bhp", "PS"])
    st.markdown("---")
    with st.expander("📊 Model Information"):
        if model_info:
            st.write(f"**Algorithm:** {model_info.get('algorithm','Unknown')}")
            st.write(f"**Training Year:** {model_info.get('current_year', 'Unknown')}")
            st.write(f"**R² Score:** {model_info.get('r2_score', 'N/A')}")
            st.write(f"**RMSE:** {model_info.get('rmse', 'N/A')}")
        else:
            st.write("No model_info loaded. Place model_info.pkl next to app.py")
    st.markdown("---")
    st.subheader("📜 Prediction History")
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df[['timestamp', 'name', 'predicted_price']].tail(5), use_container_width=True)
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet")

col_input, col_summary = st.columns([2, 1])

with col_input:
    st.header("📋 Vehicle Information")
    with st.form("prediction_form"):
        st.subheader("Basic Details")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Car Name", placeholder="e.g., Honda City VTEC", help="Display name for record keeping")
            year = st.slider("Manufacturing Year", min_value=1980, max_value=datetime.now().year, value=2020, help="Year when the car was manufactured")
            fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric', 'Hybrid', 'Other'], help="Type of fuel the vehicle uses")
            transmission = st.selectbox("Transmission", ['Manual', 'Automatic', 'Semi-Auto'], help="Transmission type")
        with col2:
            km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1000, help="Total distance the vehicle has traveled")
            seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark/Certified Dealer'], help="Type of seller")
            owner = st.selectbox("Owner Type", ['First', 'Second', 'Third', 'Fourth & Above', 'Test-Drive/Company'], help="Ownership history")
            seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9], index=2, help="Seating capacity")
        st.subheader("⚙️ Technical Specifications")
        col3, col4, col5 = st.columns(3)
        with col3:
            mileage = st.number_input(f"Mileage ({mileage_unit})", min_value=0.0, value=15.0, step=0.1, format="%.2f", help=f"Fuel efficiency in {mileage_unit}")
            engine = st.number_input("Engine Capacity (cc)", min_value=50, max_value=8000, value=1500, step=50, help="Engine displacement in cc")
        with col4:
            max_power = st.number_input(f"Max Power ({power_unit})", min_value=0.0, value=100.0, step=1.0, format="%.2f", help=f"Maximum power output in {power_unit}")
            torque_str = st.text_input("Torque", value="150 Nm @ 4000 rpm", help="Torque specification (e.g., 113 Nm @ 4000 rpm)")
        with col5:
            if include_asking_price:
                asking_price = st.number_input("Asking Price (₹)", min_value=0, value=500000, step=10000, help="Seller's listed price")
        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1,1])
        with col_btn1:
            submit = st.form_submit_button("🎯 Predict Price", type="primary")
        with col_btn2:
            reset = st.form_submit_button("🔄 Reset")

    # handle submit & reset
    if submit:
        torque_value = parse_torque(torque_str)
        input_data = {
            'name': name or "Unnamed Vehicle",
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque': torque_value if torque_value else 0,
            'seats': seats
        }
        if include_asking_price:
            input_data['asking_price'] = asking_price

        errors = validate_inputs(input_data)
        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("🔄 Analyzing vehicle data..."):
                try:
                    # API mode: send JSON and expect the same response structure
                    if use_api and api_endpoint:
                        payload = {"inputs": input_data}
                        headers = {"Content-Type": "application/json"}
                        if api_key:
                            headers['Authorization'] = f"Bearer {api_key}"
                        resp = requests.post(api_endpoint, json=payload, headers=headers, timeout=20)
                        if resp.status_code == 200:
                            resp_json = resp.json()
                            # Attempt to map/normalize the API response to our expected structure
                            result = {
                                'predicted_price': float(resp_json.get('predicted_price', resp_json.get('prediction', 0))),
                                'lower': float(resp_json.get('lower', resp_json.get('lower_bound', 0))),
                                'upper': float(resp_json.get('upper', resp_json.get('upper_bound', 0))),
                                'confidence': float(resp_json.get('confidence', 0.0)),
                                'explanation': resp_json.get('explanation', {'top_features': []})
                            }
                        else:
                            st.error(f"API request failed: {resp.status_code} {resp.text}")
                            result = mock_predict(input_data)
                    else:
                        if model is None or model_info is None:
                            st.warning("Local model not loaded — using mock predictor.")
                            result = mock_predict(input_data)
                        else:
                            # use model_info current year if available
                            result = real_predict(input_data, model, model_info, reference_df=reference_df, CURRENT_YR=model_info.get('current_year', None))
                    # store result
                    st.session_state.current_prediction = {**input_data, **result}
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'name': input_data['name'],
                        'predicted_price': f"₹{result['predicted_price']:,.0f}"
                    })
                    st.success("✅ Prediction complete!")
                    st.rerun()
                except Exception as e:
                    st.exception(f"Prediction failed: {e}")

    if reset:
        st.session_state.current_prediction = None
        st.rerun()

with col_summary:
    st.header("📊 Summary")
    if st.session_state.current_prediction:
        pred = st.session_state.current_prediction
        st.markdown(f"""
        <div class="prediction-card">
            <h2>💰 Predicted Price</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">₹{pred['predicted_price']:,.0f}</h1>
            <p style="font-size: 1.1rem;">Confidence: {pred['confidence']*100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Lower Bound", f"₹{pred['lower']:,.0f}")
        with col_r2:
            st.metric("Upper Bound", f"₹{pred['upper']:,.0f}")
        st.markdown("### 🚗 Vehicle Details")
        st.markdown(f"""
        <div class="summary-card">
            <strong>{pred['name']}</strong><br>
            📅 {pred['year']} | 🛣️ {pred['km_driven']:,} km<br>
            ⛽ {pred['fuel']} | ⚙️ {pred['transmission']}<br>
            🔑 {pred['owner']} Owner | 💺 {pred['seats']} Seats
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔍 Why this price?"):
            st.write("**Top Factors Affecting Price:**")
            for feature in pred.get('explanation', {}).get('top_features', []):
                direction_icon = "📈" if feature.get('direction') == 'positive' else "📉"
                impact = feature.get('impact', 0)
                st.markdown(f"""
                <div class="metric-container">
                    {direction_icon} <strong>{feature.get('feature')}</strong><br>
                    Impact: ₹{abs(impact):,.0f} ({feature.get('direction')})
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Fill in the vehicle details and click 'Predict Price' to see results")
        st.image("https://cdn-icons-png.flaticon.com/512/2331/2331966.png", width=200)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p><strong>Integration Instructions:</strong></p>
    <p>
    • Place your trained model as <code>car_price_model.pkl</code> in the same directory<br>
    • Place your model metadata as <code>model_info.pkl</code> in the same directory<br>
    • Optionally include <code>reference_data.pkl</code> for brand adjustments<br>
    • To use API mode: toggle "Use API for prediction" and enter your endpoint above
    </p>
    <p style="margin-top: 1rem;">Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
