import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------------- page config ----------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------- load artifacts ----------------
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("car_price_improved_model.pkl")
        enc = joblib.load("encoders_improved.pkl")
        ref = pd.read_pickle("reference_data_improved.pkl")
        return model, enc, ref
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

model, enc, ref = load_model_artifacts()

feature_cols = enc['feature_cols']
CURRENT_YR = enc['CURRENT_YR']

# Handle both old and new encoder formats
numeric_medians = enc.get('numeric_medians', {
    'Mileage_value': 18.0,
    'Engine_cc': 1200,
    'max_power': 70.0,
    'torque_value': 120.0,
    'seats': 5
})

# ---------------- helper functions ----------------
def extract_brand(name):
    parts = str(name).lower().split()
    return parts[0] if len(parts) >= 1 else "unknown"

def extract_model_name(name):
    parts = str(name).lower().split()
    return parts[1] if len(parts) >= 2 else "unknown"

def validate_inputs(year, km_driven, age, km_per_yr):
    """Validate user inputs and return warnings"""
    warnings = []
    
    if age > 30:
        warnings.append("⚠️ Very old car (30+ years) - prediction may be less accurate")
    if age < 0:
        warnings.append("⚠️ Invalid year - car appears to be from the future")
    if km_driven > 500000:
        warnings.append("⚠️ Very high mileage (500k+ km) - prediction may be less accurate")
    if km_per_yr > 40000:
        warnings.append("⚠️ Very high annual mileage (40k+ km/year) - unusual usage pattern")
    if km_per_yr < 1000 and age > 2:
        warnings.append("⚠️ Very low annual mileage - may indicate odometer issues")
    
    return warnings

def get_confidence_level(brand, model_name, bm_key, enc):
    """Determine prediction confidence based on data availability"""
    # Handle both old and new encoder formats
    if 'brand_reliable' in enc:
        brand_known = brand in enc['brand_reliable']
        model_known = model_name in enc['model_reliable']
        bm_known = bm_key in enc['bm_reliable']
    else:
        # Old format - check if exists in maps
        brand_known = brand in enc.get('brand_map_full', {})
        model_known = model_name in enc.get('model_map_full', {})
        bm_known = bm_key in enc.get('bm_map_full', {})
    
    if bm_known:
        return "High", "🟢"
    elif brand_known and model_known:
        return "Medium", "🟡"
    elif brand_known:
        return "Low", "🟠"
    else:
        return "Very Low", "🔴"

# ---------------- UI ----------------
st.title("🚗 Car Resale Price Predictor")
st.markdown("Get accurate price estimates for used cars based on advanced machine learning")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Car Details")
    
    # Basic details
    name = st.text_input(
        "Car Name (Brand Model Variant)",
        "Maruti Swift VDI",
        help="Example: Maruti Swift VDI, Hyundai i20 Sportz"
    )
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        year = st.number_input(
            "Year of Registration",
            min_value=1990,
            max_value=CURRENT_YR,
            value=2015,
            help="Year the car was first registered"
        )
    with col_b:
        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=60000,
            step=1000,
            help="Total kilometers on odometer"
        )
    with col_c:
        seats = st.number_input(
            "Number of Seats",
            min_value=2,
            max_value=10,
            value=5
        )
    
    # Transmission and fuel
    col_d, col_e = st.columns(2)
    with col_d:
        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )
    with col_e:
        fuel = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG", "LPG"]
        )
    
    # Owner and seller
    col_f, col_g = st.columns(2)
    with col_f:
        owner = st.selectbox(
            "Owner Type",
            ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Owner"]
        )
    with col_g:
        seller_type = st.selectbox(
            "Seller Type",
            ["Individual", "Dealer", "Trustmark Dealer"]
        )
    
    st.markdown("### 🔧 Technical Specifications")
    
    col_h, col_i, col_j, col_k = st.columns(4)
    with col_h:
        mileage = st.number_input(
            "Mileage (km/l)",
            min_value=0.0,
            max_value=50.0,
            value=18.0,
            step=0.1
        )
    with col_i:
        engine_cc = st.number_input(
            "Engine (CC)",
            min_value=500,
            max_value=5000,
            value=1200,
            step=50
        )
    with col_j:
        max_power = st.number_input(
            "Max Power (BHP)",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=1.0
        )
    with col_k:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=1000.0,
            value=120.0,
            step=1.0
        )

with col2:
    st.markdown("### 📊 Quick Stats")
    
    # Calculate derived values
    Age = max(CURRENT_YR - year, 0.5)
    KM_per_yr = km_driven / Age if Age > 0 else 0
    
    st.metric("Car Age", f"{Age:.1f} years")
    st.metric("Avg. KM/Year", f"{KM_per_yr:,.0f} km")
    
    # Extract brand and model
    brand = extract_brand(name)
    model_name = extract_model_name(name)
    bm_key = f"{brand}_{model_name}"
    
    # Show confidence
    confidence, icon = get_confidence_level(brand, model_name, bm_key, enc)
    st.markdown(f"**Prediction Confidence:** {icon} {confidence}")
    
    if confidence in ["Low", "Very Low"]:
        st.info("💡 Lower confidence means less training data available for this car. Prediction may vary.")

# Predict button
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])

with predict_col2:
    predict_button = st.button("🎯 PREDICT PRICE", type="primary", use_container_width=True)

if predict_button:
    # Validate inputs
    validation_warnings = validate_inputs(year, km_driven, Age, KM_per_yr)
    
    if validation_warnings:
        with st.expander("⚠️ Input Warnings", expanded=True):
            for warning in validation_warnings:
                st.warning(warning)
    
    # ---------------- feature engineering ----------------
    # Age and KM calculations
    Age = max(CURRENT_YR - year, 0.5)
    if Age < 0:
        Age = 0.5
    
    KM_per_yr = km_driven / Age if Age > 0 else 0
    KM_per_yr = min(KM_per_yr, 50000)  # Cap extreme values
    
    # Encode categorical variables
    transmission_encoded = 0 if transmission == "Manual" else 1
    
    diesel = 1 if fuel == "Diesel" else 0
    petrol = 1 if fuel == "Petrol" else 0
    lpg = 1 if fuel == "LPG" else 0
    cng = 1 if fuel == "CNG" else 0
    
    o1 = 1 if owner == "First Owner" else 0
    o2 = 1 if owner == "Second Owner" else 0
    o3 = 1 if owner == "Third Owner" else 0
    o4 = 1 if owner == "Fourth & Above Owner" else 0
    o_test = 1 if owner == "Test Drive Owner" else 0
    
    ind = 1 if seller_type == "Individual" else 0
    dea = 1 if seller_type == "Dealer" else 0
    trust = 1 if seller_type == "Trustmark Dealer" else 0
    
    # Additional features
    power_to_weight = max_power / (engine_cc / 1000) if engine_cc > 0 else 0
    Age_squared = Age ** 2
    km_driven_log = np.log1p(km_driven)
    
    luxury_brands = ['bmw', 'audi', 'mercedes-benz', 'jaguar', 'land', 'porsche']
    budget_brands = ['maruti', 'tata', 'hyundai', 'datsun']
    is_luxury = 1 if brand in luxury_brands else 0
    is_budget = 1 if brand in budget_brands else 0
    
    # ---------------- target encodings ----------------
    # Handle both old and new encoder formats
    if 'brand_map' in enc:
        # New format
        brand_te = enc['brand_map'].get(brand, enc['brand_global'])
        model_te = enc['model_map'].get(model_name, enc['model_global'])
        bm_te = enc['bm_map'].get(bm_key, enc['bm_global'])
    else:
        # Old format
        brand_te = enc.get('brand_map_full', {}).get(brand, enc.get('brand_global', 300000))
        model_te = enc.get('model_map_full', {}).get(model_name, enc.get('model_global', 300000))
        bm_te = enc.get('bm_map_full', {}).get(bm_key, enc.get('bm_global', 300000))
    
    # ---------------- build feature row ----------------
    # Start with base features that should always exist
    row = {
        'Age': Age,
        'km_driven': km_driven,
        'KM_per_yr': KM_per_yr,
        'transmission': transmission_encoded,  # Old format
        'transmission_encoded': transmission_encoded,  # New format

    }