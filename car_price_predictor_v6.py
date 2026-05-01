import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    from rapidfuzz import process as fuzz_process, fuzz
    FUZZY_OK = True
except ImportError:
    FUZZY_OK = False

# ================================================================
# UTIL FUNCTIONS
# ================================================================

def extract_first_number(x):
    if pd.isna(x): return np.nan
    x = str(x).replace(',', '')
    match = re.search(r"\d+\.?\d*", x)
    return float(match.group()) if match else np.nan

def extract_engine_cc(s):
    v = extract_first_number(s)
    return int(v) if not pd.isna(v) else np.nan

SKIP_WORDS = {'suzuki','motors','rover','benz','royce','motor','renault'}

def extract_brand(name):
    if pd.isna(name): return "unknown"
    return str(name).lower().strip().split()[0]

def extract_model(name):
    if pd.isna(name): return "unknown"
    parts = str(name).lower().strip().split()
    if len(parts) < 2: return "unknown"
    if len(parts) >= 3 and parts[1] in SKIP_WORDS:
        return " ".join(parts[2:])
    return " ".join(parts[1:])

# ================================================================
# EV DETECTION
# ================================================================

EV_KEYWORDS = {
    'ev', 'electric', 'nexon ev', 'zs ev', 'tigor ev', 'tiago ev',
    'e-tron', 'etron', 'id.4', 'id4', 'leaf', 'ioniq', 'kona electric',
    'mg zs ev', 'bz4x', 'nexon.ev', 'xev', 'e20'
}

def is_electric(name, fuel):
    name_lower = str(name).lower()
    # Check name contains EV keywords
    if any(kw in name_lower for kw in EV_KEYWORDS):
        return True
    # Check fuel explicitly set to electric
    if str(fuel).lower() in ('electric', 'ev'):
        return True
    return False

# ================================================================
# DEPRECIATION TABLES
# ================================================================

# Standard ICE cars (Petrol/Diesel/CNG/LPG)
# Real Indian used car market — NOT insurance IDV
# Based on actual transaction data from CarDekho/OLX/Cars24
ICE_DEPRECIATION = {
    0: 0.00, 1: 0.15, 2: 0.28, 3: 0.38,
    4: 0.45, 5: 0.50, 6: 0.53, 7: 0.55,
    8: 0.57, 9: 0.58, 10: 0.60,
}

# EVs depreciate faster in India due to:
# - Battery health anxiety among buyers
# - Charging infrastructure uncertainty
# - Rapid technology change (older EVs feel outdated fast)
# - Limited resale demand outside metro cities
EV_DEPRECIATION = {
    0: 0.00, 1: 0.20, 2: 0.35, 3: 0.45,
    4: 0.52, 5: 0.58, 6: 0.63, 7: 0.67,
    8: 0.70, 9: 0.72, 10: 0.74,
}

# Luxury ICE cars (BMW/Audi/Mercedes/Jaguar etc.)
# Depreciate faster than mainstream in India
# High maintenance cost scares buyers → lower demand
LUXURY_DEPRECIATION = {
    0: 0.00, 1: 0.20, 2: 0.35, 3: 0.47,
    4: 0.55, 5: 0.62, 6: 0.67, 7: 0.70,
    8: 0.72, 9: 0.74, 10: 0.76,
}

LUXURY_BRANDS_SET = {'bmw','audi','mercedes-benz','jaguar','land','porsche','volvo','lexus'}

def get_depreciation(age: float, car_name: str, fuel: str, brand: str) -> tuple:
    """
    Returns (depreciation_fraction, table_used_label)
    Picks the right depreciation table based on car type.
    """
    if is_electric(car_name, fuel):
        table = EV_DEPRECIATION
        label = "EV"
    elif brand.lower() in LUXURY_BRANDS_SET:
        table = LUXURY_DEPRECIATION
        label = "Luxury"
    else:
        table = ICE_DEPRECIATION
        label = "ICE"

    age_int = min(int(np.floor(age)), max(table.keys()))
    frac    = age - int(np.floor(age))
    base    = table[age_int]
    nxt     = table.get(age_int + 1, base)
    depr    = base + frac * (nxt - base)
    return depr, label

# Keep a simple wrapper for training feature
def irda_depreciation(age: float) -> float:
    table   = ICE_DEPRECIATION
    age_int = min(int(np.floor(age)), max(table.keys()))
    frac    = age - int(np.floor(age))
    base    = table[age_int]
    nxt     = table.get(age_int + 1, base)
    return base + frac * (nxt - base)

def apply_market_correction(ml_price, purchase_price, age, km,
                             car_name, fuel, brand):
    depr_rate, depr_label = get_depreciation(age, car_name, fuel, brand)
    market_value = purchase_price * (1 - depr_rate)

    # ML weight: trust ML more for recent cars, market curve more for older
    ml_weight = max(0.35, 1.0 - age * 0.08)

    # For EVs: trust market curve more (ML has almost no EV training data)
    if depr_label == "EV":
        ml_weight = max(0.15, ml_weight - 0.20)

    blended = ml_weight * ml_price + (1 - ml_weight) * market_value

    # KM penalty: every 10K km over expected (15K/yr) → -0.5%
    expected_km = age * 15000
    excess_km   = max(0, km - expected_km)
    km_penalty  = max(0.80, 1 - (excess_km / 10000) * 0.005)

    # Low KM bonus: under-driven cars worth slightly more
    if km < expected_km * 0.6:
        km_penalty = min(1.05, km_penalty + 0.03)

    final = blended * km_penalty
    return {
        "ml_raw":      int(ml_price),
        "market_floor": int(market_value),
        "blended":     int(blended),
        "km_penalty":  round(km_penalty, 3),
        "final":       int(final),
        "depr_pct":    round(depr_rate * 100, 1),
        "depr_label":  depr_label,
        "ml_weight":   round(ml_weight, 2),
    }

# ================================================================
# FUZZY MATCH
# ================================================================

def fuzzy_match_variant(brand_key, model_key, bm_map, top_n=6, threshold=55):
    if not FUZZY_OK:
        candidates = [k for k in bm_map if brand_key in k]
        scored = []
        model_words = set(model_key.split())
        for c in candidates:
            c_words = set(c.split('_', 1)[-1].split())
            overlap = len(model_words & c_words) / max(len(model_words), 1)
            scored.append((c, int(overlap * 100)))
        scored.sort(key=lambda x: -x[1])
        return [(k, s) for k, s in scored[:top_n] if s >= 20]

    candidates = {k: k.split('_', 1)[-1] for k in bm_map if brand_key in k}
    if not candidates:
        return []
    results = fuzz_process.extract(
        model_key, candidates,
        scorer=fuzz.token_set_ratio, limit=top_n
    )
    return [(r[2], r[1]) for r in results if r[1] >= threshold]

def prompt_user_variant(brand_key, model_key, bm_map):
    matches = fuzzy_match_variant(brand_key, model_key, bm_map)
    if not matches:
        print(f"     ⚠️  No variants found for '{brand_key}' in dataset.")
        print(f"        Using brand median as fallback.")
        return None

    exact_bm = brand_key + "_" + model_key
    if exact_bm in bm_map:
        return exact_bm

    print(f"\n  ⚠️  Exact variant not found. Closest matches in dataset:")
    print(f"  {'#':<3} {'Variant':<45} {'Match%':>7}  {'Median Price':>13}")
    print(f"  {'─'*3} {'─'*45} {'─'*7}  {'─'*13}")
    for i, (bm_key_candidate, score) in enumerate(matches, 1):
        model_part = bm_key_candidate.split('_', 1)[-1]
        price      = bm_map.get(bm_key_candidate, 0)
        print(f"  {i:<3} {model_part:<45} {int(score):>6}%  ₹{price:>12,.0f}")

    print(f"\n  Enter number to select, or 0 to use brand fallback: ", end="")
    try:
        choice = int(input().strip())
        if 1 <= choice <= len(matches):
            chosen = matches[choice - 1][0]
            print(f"  ✅ Using: '{chosen.split('_',1)[-1]}'")
            return chosen
    except (ValueError, IndexError):
        pass
    print(f"  Using brand median fallback.")
    return None

# ================================================================
# K-FOLD SMOOTHED TARGET ENCODING
# ================================================================

def kfold_target_encode(train_df, test_df, col, target, n_splits=5, smoothing=10):
    global_mean = train_df[target].mean()
    train_enc   = np.zeros(len(train_df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        fold_tr  = train_df.iloc[tr_idx]
        fold_val = train_df.iloc[val_idx]
        stats = fold_tr.groupby(col)[target].agg(['mean', 'count'])
        stats['smoothed'] = (
            (stats['mean'] * stats['count'] + global_mean * smoothing)
            / (stats['count'] + smoothing)
        )
        train_enc[val_idx] = fold_val[col].map(stats['smoothed']).fillna(global_mean).values
    stats_all = train_df.groupby(col)[target].agg(['mean', 'count'])
    stats_all['smoothed'] = (
        (stats_all['mean'] * stats_all['count'] + global_mean * smoothing)
        / (stats_all['count'] + smoothing)
    )
    test_enc = test_df[col].map(stats_all['smoothed']).fillna(global_mean).values
    return train_enc, test_enc, stats_all['smoothed'].to_dict(), global_mean

# ================================================================
# LOAD & CLEAN
# ================================================================

df = pd.read_csv('Car details v3.csv')

df['year']      = pd.to_numeric(df['year'],      errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
df['seats']     = pd.to_numeric(df['seats'],     errors='coerce')

CURRENT_YR = 2026

df['Age']  = (CURRENT_YR - df['year']).astype(float)
df.loc[df['Age'] <= 0, 'Age'] = 0.5

df['KM_per_yr']           = df['km_driven'] / df['Age']
df['Age_squared']         = df['Age'] ** 2
df['km_driven_log']       = np.log1p(df['km_driven'])
df['depreciation_factor'] = df['Age'].apply(lambda a: 1 - irda_depreciation(a))
df.drop(columns=['year'], inplace=True)

df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})

df['fuel'] = df['fuel'].fillna('Unknown')
df['diesel'] = (df['fuel'] == 'Diesel').astype(int)
df['petrol'] = (df['fuel'] == 'Petrol').astype(int)
df['lpg']    = (df['fuel'] == 'LPG').astype(int)
df['cng']    = (df['fuel'] == 'CNG').astype(int)

df['owner'] = df['owner'].fillna('Unknown')
df['1st own']        = (df['owner'] == 'First Owner').astype(int)
df['2nd own']        = (df['owner'] == 'Second Owner').astype(int)
df['3rd own']        = (df['owner'] == 'Third Owner').astype(int)
df['4+ own']         = (df['owner'] == 'Fourth & Above Owner').astype(int)
df['test drive own'] = (df['owner'] == 'Test Drive Owner').astype(int)

df['seller_type'] = df['seller_type'].fillna('Unknown')
df['individual']       = (df['seller_type'] == 'Individual').astype(int)
df['dealer']           = (df['seller_type'] == 'Dealer').astype(int)
df['trustmark dealer'] = (df['seller_type'] == 'Trustmark Dealer').astype(int)

df['Mileage_value'] = df['mileage'].apply(extract_first_number)
df['Engine_cc']     = df['engine'].apply(extract_engine_cc)
df['max_power']     = df['max_power'].apply(extract_first_number)
df['torque_value']  = df['torque'].apply(extract_first_number)

df['brand']       = df['name'].apply(extract_brand)
df['model']       = df['name'].apply(extract_model)
df['brand_model'] = df['brand'] + "_" + df['model']

luxury_brands = list(LUXURY_BRANDS_SET)
df['is_luxury'] = df['brand'].isin(luxury_brands).astype(int)

df['power_to_weight']    = (df['max_power'] / (df['Engine_cc'] / 1000)
                            ).replace([np.inf,-np.inf], 0).fillna(0)
df['age_km_interaction'] = df['Age'] * df['km_driven']

print("\n📋 Extraction check (first 5 rows):")
print(df[['name','brand','model','brand_model']].head())

# ================================================================
# FEATURES
# ================================================================

base_features = [
    'Age', 'Age_squared', 'km_driven', 'km_driven_log', 'KM_per_yr',
    'depreciation_factor', 'age_km_interaction',
    'transmission',
    'diesel', 'petrol', 'lpg', 'cng',
    '1st own', '2nd own', '3rd own', '4+ own', 'test drive own',
    'individual', 'dealer', 'trustmark dealer',
    'Mileage_value', 'Engine_cc', 'max_power', 'torque_value', 'seats',
    'power_to_weight', 'is_luxury',
]
feature_col = base_features + ['brand_te', 'model_te', 'bm_te']

df = df.dropna(subset=base_features + ['selling_price']).reset_index(drop=True)

# ================================================================
# SPLIT → ENCODE → TRAIN
# ================================================================

y = np.log1p(df['selling_price'])
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

train_df = df.loc[train_idx].reset_index(drop=True)
test_df  = df.loc[test_idx].reset_index(drop=True)
y_train  = y.loc[train_idx].reset_index(drop=True)
y_test   = y.loc[test_idx].reset_index(drop=True)

brand_tr, brand_te_v, brand_map, global_mean = kfold_target_encode(train_df, test_df, 'brand',       'selling_price')
model_tr, model_te_v, model_map, _           = kfold_target_encode(train_df, test_df, 'model',       'selling_price')
bm_tr,    bm_te_v,    bm_map,    _           = kfold_target_encode(train_df, test_df, 'brand_model', 'selling_price')

brand_median_map = train_df.groupby('brand')['selling_price'].median().to_dict()

train_df = train_df.copy(); test_df = test_df.copy()
for frame, b, m, bm in [(train_df, brand_tr, model_tr, bm_tr),
                         (test_df,  brand_te_v, model_te_v, bm_te_v)]:
    frame['brand_te'] = b
    frame['model_te'] = m
    frame['bm_te']    = bm

X_train = train_df[feature_col]
X_test  = test_df[feature_col]

xgb = XGBRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=3.0, min_child_weight=10, gamma=0.1,
    random_state=42, early_stopping_rounds=50, eval_metric='rmse',
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# ================================================================
# EVAL
# ================================================================

y_pred        = np.expm1(xgb.predict(X_test))
y_test_actual = np.expm1(y_test)

print(f"\n{'='*45}")
print(f"  R²   : {r2_score(y_test_actual, y_pred):.4f}")
print(f"  MAE  : ₹{mean_absolute_error(y_test_actual, y_pred):,.0f}")
print(f"  RMSE : ₹{np.sqrt(mean_squared_error(y_test_actual, y_pred)):,.0f}")
print(f"{'='*45}\n")

# ================================================================
# SAVE
# ================================================================

joblib.dump(xgb, "car_price_model.pkl")
joblib.dump({
    "brand_map": brand_map, "model_map": model_map, "bm_map": bm_map,
    "global_mean": global_mean, "feature_cols": feature_col,
    "CURRENT_YR": CURRENT_YR, "luxury_brands": luxury_brands,
    "brand_median_map": brand_median_map,
}, "encoders.pkl")
print("✅ Model and encoders saved.\n")

if not FUZZY_OK:
    print("⚠️  rapidfuzz not installed — fuzzy matching disabled.")
    print("   Run:  pip install rapidfuzz\n")

# ================================================================
# PREDICTION
# ================================================================

def predict_car():
    print("\n🚗  Car Resale Price Estimator  (India)  v6")
    print("-" * 48)
    print("  💡 Tip: type full variant name for best accuracy")
    print("     e.g. 'Hyundai Elite i20 Petrol Sportz BSIV'")
    print("-" * 48)

    name           = input("Car Name                                    : ").strip()
    year           = int(input("Year of Purchase                            : ") or 2015)
    km             = float(input("KM Driven                                   : ") or 60000)
    purchase_price = float(input("Ex-Showroom Price when bought (₹)           : ") or 500000)
    trans_in       = input("Transmission  (Manual / Automatic)          : ").strip()
    fuel_in        = input("Fuel  (Petrol / Diesel / CNG / LPG)         : ").strip().upper()
    owner_in       = input("Owner (First Owner / Second Owner /\n"
                           "       Third Owner / Fourth & Above Owner\n"
                           "       / Test Drive Owner)                  : ").strip()
    seller_in      = input("Seller (Individual / Dealer /\n"
                           "        Trustmark Dealer)                   : ").strip()
    mileage        = float(input("Mileage (kmpl / km per charge for EV)       : ") or 20)
    engine         = int(input("Engine CC  (for EV enter 0)                 : ") or 0)
    power          = float(input("Max Power (bhp)                             : ") or 80)
    torque         = float(input("Torque (Nm)                                 : ") or 100)
    seats          = int(input("Seats                                       : ") or 5)

    Age = float(CURRENT_YR - year)
    if Age <= 0: Age = 0.5

    brand_key = extract_brand(name)
    model_key = extract_model(name)
    bm_key    = brand_key + "_" + model_key

    # Auto-detect EV and show notice
    ev_detected = is_electric(name, fuel_in)
    if ev_detected:
        print(f"\n  ⚡ EV detected — applying EV depreciation curve")
        print(f"     (EVs depreciate faster in India: battery anxiety + limited resale demand)")

    chosen_bm = prompt_user_variant(brand_key, model_key, bm_map)

    if chosen_bm:
        final_bm_te    = bm_map[chosen_bm]
        chosen_model   = chosen_bm.split('_', 1)[-1]
        final_model_te = model_map.get(chosen_model,
                         brand_median_map.get(brand_key, global_mean))
        match_label = "fuzzy match ✅" if chosen_bm != bm_key else "exact match ✅"
    else:
        final_bm_te    = brand_median_map.get(brand_key, global_mean)
        final_model_te = brand_median_map.get(brand_key, global_mean)
        match_label    = "brand median fallback"

    final_brand_te = brand_map.get(brand_key, global_mean)

    print(f"\n  🔍 brand='{brand_key}' | model='{model_key}'")
    print(f"     brand_te : ₹{final_brand_te:,.0f}")
    print(f"     model_te : ₹{final_model_te:,.0f}  ({match_label})")
    print(f"     bm_te    : ₹{final_bm_te:,.0f}")

    # Engine CC for EV: use 0 or small value, won't affect much
    engine_cc = engine if engine > 0 else 1   # avoid div/0

    row = {
        'Age': Age, 'Age_squared': Age**2,
        'km_driven': km, 'km_driven_log': np.log1p(km),
        'KM_per_yr': km / Age,
        'depreciation_factor': 1 - irda_depreciation(Age),
        'age_km_interaction': Age * km,
        'transmission': 0 if trans_in.lower().startswith('m') else 1,
        'diesel': int(fuel_in == 'DIESEL'), 'petrol': int(fuel_in == 'PETROL'),
        'lpg':    int(fuel_in == 'LPG'),    'cng':    int(fuel_in == 'CNG'),
        '1st own':        int(owner_in == 'First Owner'),
        '2nd own':        int(owner_in == 'Second Owner'),
        '3rd own':        int(owner_in == 'Third Owner'),
        '4+ own':         int(owner_in == 'Fourth & Above Owner'),
        'test drive own': int(owner_in == 'Test Drive Owner'),
        'individual':       int(seller_in == 'Individual'),
        'dealer':           int(seller_in == 'Dealer'),
        'trustmark dealer': int(seller_in == 'Trustmark Dealer'),
        'Mileage_value': mileage, 'Engine_cc': engine_cc,
        'max_power': power, 'torque_value': torque, 'seats': seats,
        'power_to_weight': power / (engine_cc / 1000),
        'is_luxury': int(brand_key in LUXURY_BRANDS_SET),
        'brand_te':  final_brand_te,
        'model_te':  final_model_te,
        'bm_te':     final_bm_te,
    }

    X_user  = pd.DataFrame([row])[feature_col]
    ml_pred = np.expm1(xgb.predict(X_user)[0])
    result  = apply_market_correction(
        ml_pred, purchase_price, Age, km, name, fuel_in, brand_key
    )

    depr_label = result['depr_label']
    depr_icon  = {'EV': '⚡', 'Luxury': '💎', 'ICE': '🔥'}.get(depr_label, '🔥')

    print(f"\n{'='*48}")
    print(f"  🤖  ML Raw Estimate         : ₹{result['ml_raw']:>10,}")
    print(f"  {depr_icon}  Market Floor ({depr_label:<7})    : ₹{result['market_floor']:>10,}  ({result['depr_pct']}% depr.)")
    print(f"  🔀  Blend ({result['ml_weight']:.0%} ML + {1-result['ml_weight']:.0%} market)  : ₹{result['blended']:>10,}")
    print(f"  🛣️   KM Adjustment           :   {result['km_penalty']:.1%}")
    print(f"  {'─'*46}")
    print(f"  💰  FINAL Resale Estimate   : ₹{result['final']:>10,}")
    print(f"{'='*48}")
    low  = int(result['final'] * 0.92)
    high = int(result['final'] * 1.08)
    print(f"\n  📊 Realistic Range : ₹{low:,}  –  ₹{high:,}")
    print(f"     (±8% for condition / negotiation)\n")

    # Confidence indicator
    if match_label == "exact match ✅":
        conf = "HIGH"
        conf_note = "exact variant found in dataset"
    elif match_label == "fuzzy match ✅":
        conf = "MEDIUM"
        conf_note = "closest variant used — verify specs match"
    else:
        conf = "LOW"
        conf_note = "brand median used — result is approximate"

    if ev_detected:
        conf = "LOW" if conf == "HIGH" else conf
        conf_note += " | EV not in training dataset"

    conf_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "🟡")
    print(f"  {conf_icon} Confidence: {conf}  ({conf_note})\n")

predict_car()
