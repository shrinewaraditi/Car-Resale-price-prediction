# diagnose_prediction.py
import joblib, pandas as pd, numpy as np
from pathlib import Path

MODEL = "car_price_improved_model.pkl"
ENC = "encoders_improved.pkl"
REF = "reference_data_improved.pkl"

# example input (same as you tested)
user = {
  'name': "Hyundai i20 Sportz",
  'year': 2017,
  'km_driven': 45000,
  'transmission': 'Manual',
  'fuel': 'Petrol',
  'owner': 'First',
  'seller_type': 'Individual',
  'mileage': 18.5,
  'engine': 1197,
  'max_power': 83.0,
  'torque': 114.0,
  'seats': 5,
  'condition': 'Good'
}

def load(path):
    p = Path(path)
    if not p.exists():
        print(f"Missing: {path}")
        return None
    return joblib.load(path)

m = load(MODEL)
enc = load(ENC)
ref = None
try:
    ref = pd.read_pickle(REF)
except Exception as e:
    print("Could not load ref df:", e)

print("\n=== encoder keys ===")
if isinstance(enc, dict):
    for k in sorted(enc.keys()):
        print(k)
else:
    print("encoders not dict:", type(enc))

# inspect expected feature cols
feature_cols = None
if isinstance(enc, dict):
    feature_cols = enc.get('feature_cols') or enc.get('feature_columns')
print("\nExpected feature cols length:", None if feature_cols is None else len(feature_cols))
print("First 40 feature cols preview:", feature_cols[:40] if feature_cols else feature_cols)

# Build X_user like the app does
CURRENT_YR = enc.get('CURRENT_YR') if isinstance(enc, dict) else 2025
year = user['year']
Age = CURRENT_YR - year
if Age == 0: Age = 0.5
KM_per_yr = user['km_driven'] / Age if Age>0 else 0.0

# basic one-hot as app
transmission_encoded = 0 if user['transmission'].lower().startswith('m') else 1
diesel = 1 if user['fuel']=='Diesel' else 0
petrol = 1 if user['fuel']=='Petrol' else 0
lpg = 1 if user['fuel']=='LPG' else 0
cng = 1 if user['fuel']=='CNG' else 0
o1 = 1 if str(user['owner']).startswith('First') else 0
o2 = 1 if str(user['owner']).startswith('Second') else 0
o3 = 1 if str(user['owner']).startswith('Third') else 0
o4 = 1 if 'Fourth' in str(user['owner']) else 0
o_test = 1 if ('Test' in str(user['owner']) or 'Company' in str(user['owner'])) else 0
ind = 1 if user['seller_type']=='Individual' else 0
dea = 1 if user['seller_type']=='Dealer' else 0
trust = 1 if 'Trust' in user['seller_type'] else 0

feature_dict = {
    'Age': Age, 'km_driven': user['km_driven'], 'KM_per_yr': KM_per_yr,
    'transmission': transmission_encoded,
    'diesel': diesel, 'petrol': petrol, 'lpg': lpg, 'cng': cng,
    '1st own': o1, '2nd own': o2, '3rd own': o3, '4+ own': o4, 'test drive own': o_test,
    'individual': ind, 'dealer': dea, 'trustmark dealer': trust,
    'Mileage_value': user['mileage'], 'Engine_cc': user['engine'],
    'max_power': user['max_power'], 'torque_value': user['torque'], 'seats': user['seats']
}

X_user = pd.DataFrame([feature_dict])
print("\nConstructed base X_user (pre-encodings):")
print(X_user.T)

# If enc contains brand/model maps, compute encoded features
brand_map = enc.get('brand_map_full') if isinstance(enc, dict) else None
brand_global = enc.get('brand_global') if isinstance(enc, dict) else None
model_map = enc.get('model_map_full') if isinstance(enc, dict) else None
model_global = enc.get('model_global') if isinstance(enc, dict) else None
bm_map = enc.get('bm_map_full') if isinstance(enc, dict) else None
bm_global = enc.get('bm_global') if isinstance(enc, dict) else None

brand = user['name'].split()[0].lower()
model_name = user['name'].split()[1].lower() if len(user['name'].split())>1 else 'unknown'
bm_key = f"{brand}_{model_name}"

print("\nbrand token:", brand, "model token:", model_name)

def get_map_val(mapping, key, global_val):
    if mapping is None:
        return None
    return mapping.get(key, global_val)

brand_te = get_map_val(brand_map, brand, brand_global)
model_te = get_map_val(model_map, model_name, model_global)
bm_te = get_map_val(bm_map, bm_key, bm_global)

print("brand_te (from encoders):", brand_te)
print("model_te (from encoders):", model_te)
print("brand_model_te (from encoders):", bm_te)

# If the feature_cols includes these encoded names, append to X_user
encoded_names = []
if feature_cols:
    for name in ['brand_te','model_te','brand_model_te','Age_x_brand','KMpy_x_brand','Age_x_bm']:
        if name in feature_cols and name not in X_user.columns:
            encoded_names.append(name)

print("\nEncoded names expected by feature_cols present but missing in base X_user:", encoded_names)

# Fill encoded features (fallback to brand medians from ref df if enc maps missing)
if 'brand_te' in encoded_names:
    if brand_te is None and isinstance(ref, pd.DataFrame):
        # fallback brand median
        names_lower = ref['name'].dropna().astype(str).str.lower()
        if any(names_lower.str.startswith(brand)):
            brand_te = ref[names_lower.str.startswith(brand)]['selling_price'].median()
    X_user['brand_te'] = brand_te if brand_te is not None else 0.0

if 'model_te' in encoded_names:
    if model_te is None and isinstance(ref, pd.DataFrame):
        names_lower = ref['name'].dropna().astype(str).str.lower()
        if any(names_lower.str.contains(model_name)):
            model_te = ref[names_lower.str.contains(model_name)]['selling_price'].median()
    X_user['model_te'] = model_te if model_te is not None else 0.0

if 'brand_model_te' in encoded_names:
    if bm_te is None and isinstance(ref, pd.DataFrame):
        names_lower = ref['name'].dropna().astype(str).str.lower()
        mask = names_lower.str.startswith(brand) & names_lower.str.contains(model_name)
        if mask.any():
            bm_te = ref.loc[mask, 'selling_price'].median()
    X_user['brand_model_te'] = bm_te if bm_te is not None else 0.0

# Add interaction columns if expected
if 'Age_x_brand' in encoded_names:
    X_user['Age_x_brand'] = X_user['Age'] * X_user.get('brand_te', 0.0)
if 'KMpy_x_brand' in encoded_names:
    X_user['KMpy_x_brand'] = X_user['KM_per_yr'] * X_user.get('brand_te', 0.0)
if 'Age_x_bm' in encoded_names:
    X_user['Age_x_bm'] = X_user['Age'] * X_user.get('brand_model_te', 0.0)

print("\nFinal X_user with encoded cols (columns shown):")
print(X_user.T)

# Reorder to feature_cols if available
if feature_cols:
    X_user = X_user.reindex(columns=feature_cols, fill_value=0.0)

print("\nX_user aligned to model columns (first 40 cols):")
print(X_user.columns[:40])
print(X_user.iloc[0].to_dict())

# Predict
if m is not None:
    try:
        raw_pred = float(m.predict(X_user)[0])
        
        
        print(f"\nRaw model output: {raw_pred}")
    except Exception as e:
        print("Model prediction failed:", e)

    # blending
    blended = raw_pred
    if isinstance(ref, pd.DataFrame):
        names_lower = ref['name'].dropna().astype(str).str.lower()
        if any(names_lower.str.startswith(brand)):
            brand_med = ref[names_lower.str.startswith(brand)]['selling_price'].median()
            print("Brand median:", brand_med)
            blended = 0.5*raw_pred + 0.5*brand_med
            print("Blended (50/50):", blended)

    # condition
    cond = user.get('condition','Good').lower()
    cond_map = {'excellent':1.05, 'good':1.0, 'average':0.95, 'below':0.9, 'poor':0.85}
    final = blended * cond_map.get(cond,1.0)
    print("Final after condition:", final)
else:
    print("Model not loaded.")
