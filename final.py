import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------------- UTIL FUNCTIONS ----------------

def extract_number(s):
    if pd.isna(s): return np.nan
    s = str(s).replace(',', '')
    match = re.search(r"\d+\.?\d*", s)
    return float(match.group()) if match else np.nan
import re
import pandas as pd
import numpy as np
import joblib  # Add this import
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------- Load dataset -------------------
df = pd.read_csv('Car details v3.csv')

def extract_first_number(s):
    if pd.isna(s): return np.nan
    s = str(s).replace(',', '')
    m = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(m.group()) if m else np.nan

def extract_engine_cc(s):
    v = extract_first_number(s)
    return int(v) if not pd.isna(v) else np.nan

def extract_brand(name):
    if pd.isna(name): return "unknown"
    return str(name).split()[0].lower()

# ------------------- Data Cleaning -------------------
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
df['seats'] = pd.to_numeric(df['seats'], errors='coerce')

CURRENT_YR = 2025
df['Age'] = CURRENT_YR - df['year'].astype(float)
df.loc[df['Age'] == 0, 'Age'] = 0.5
df['KM_per_yr'] = df['km_driven'] / df['Age']
df.drop(columns=['year'], inplace=True)

# ------------------- Encode categorical features -------------------
df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})
df['fuel'] = df['fuel'].fillna('Unknown').astype(str)
df['diesel'] = (df['fuel'] == 'Diesel').astype(int)
df['petrol'] = (df['fuel'] == 'Petrol').astype(int)
df['lpg'] = (df['fuel'] == 'LPG').astype(int)
df['cng'] = (df['fuel'] == 'CNG').astype(int)

df['owner'] = df['owner'].fillna('Unknown').astype(str)
df['1st own'] = (df['owner'] == 'First Owner').astype(int)
df['2nd own'] = (df['owner'] == 'Second Owner').astype(int)
df['3rd own'] = (df['owner'] == 'Third Owner').astype(int)
df['4+ own'] = (df['owner'] == 'Fourth & Above Owner').astype(int)
df['test drive own'] = (df['owner'] == 'Test Drive Owner').astype(int)

df['seller_type'] = df['seller_type'].fillna('Unknown').astype(str)
df['individual'] = (df['seller_type'] == 'Individual').astype(int)
df['dealer'] = (df['seller_type'] == 'Dealer').astype(int)
df['trustmark dealer'] = (df['seller_type'] == 'Trustmark Dealer').astype(int)

df['Mileage_value'] = df['mileage'].apply(extract_first_number)
df['Engine_cc'] = df['engine'].apply(extract_engine_cc)
df['max_power'] = df['max_power'].apply(extract_first_number)
df['torque_value'] = df['torque'].apply(extract_first_number)
df['brand'] = df['name'].apply(extract_brand)

feature_col = ['Age', 'km_driven', 'KM_per_yr', 'transmission',
               'diesel', 'petrol', 'lpg', 'cng',
               '1st own', '2nd own', '3rd own', '4+ own', 'test drive own',
               'individual', 'dealer', 'trustmark dealer',
               'Mileage_value', 'Engine_cc', 'max_power', 'torque_value', 'seats']

for c in feature_col:
    if c not in df.columns:
        df[c] = 0.0

X = df[feature_col].copy()
y = pd.to_numeric(df['selling_price'].copy(), errors='coerce')

# ------------------- Train Model -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=500, max_depth=32, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 score {r2 * 100:.2f}")
print(f"mae {mae:.2f}")
print(f"rmse {rmse:.2f}\n")

# ------------------- 🆕 SAVE THE MODEL -------------------
print("\n" + "="*60)
print("💾 Saving model and metadata...")
print("="*60)

# Save the trained model
joblib.dump(model, 'car_price_model.pkl')
print("✅ Model saved as 'car_price_model.pkl'")

# Save feature information and metrics
feature_info = {
    'feature_columns': feature_col,
    'current_year': CURRENT_YR,
    'r2_score': r2,
    'mae': mae,
    'rmse': rmse,
    'n_estimators': 500,
    'max_depth': 32
}
joblib.dump(feature_info, 'model_info.pkl')
print("✅ Model info saved as 'model_info.pkl'")

# Save a copy of the reference dataframe for brand adjustments
df.to_pickle('reference_data.pkl')
print("✅ Reference data saved as 'reference_data.pkl'")

print("\n✨ All files saved successfully!")
print("Files created:")
print("  1. car_price_model.pkl")
print("  2. model_info.pkl")
print("  3. reference_data.pkl")
print("\n📂 Place these files in the same directory as your Streamlit app")
print("="*60)

# ------------------- Prediction Function -------------------
def prompt_input_predict_adjusted(model, df, CURRENT_YR=2025):
    print("*" * 65)
    print(" " * 20 + "🚗 Enter your Car Details 🚗" + " " * 20)
    print("*" * 65)

    year = int(input("Year of registration (e.g. 2015) [default 2015]: ") or 2015)
    km_driven = float(input("Total km driven (e.g. 60000) [default 60000]: ") or 60000)
    trans_input = input("Transmission (Manual/Automatic) [default Manual]: ") or "Manual"
    transmission = 0 if trans_input.lower().startswith('m') else 1
    fuel_input = (input("Fuel (Diesel/Petrol/CNG/LPG/Other) [default Diesel]: ") or "Diesel").title()
    diesel = 1 if fuel_input == 'Diesel' else 0
    petrol = 1 if fuel_input == 'Petrol' else 0
    cng = 1 if fuel_input == 'CNG' else 0
    lpg = 1 if fuel_input == 'LPG' else 0
    owner_input = (input("Owner (First Owner/Second Owner/Third Owner/Fourth & Above Owner/Test Drive Owner) [default First Owner]: ") or "First Owner")
    o1 = 1 if owner_input == 'First Owner' else 0
    o2 = 1 if owner_input == 'Second Owner' else 0
    o3 = 1 if owner_input == 'Third Owner' else 0
    o4 = 1 if owner_input == 'Fourth & Above Owner' else 0
    o_test = 1 if owner_input == 'Test Drive Owner' else 0
    seller_input = (input("Seller type (Individual/Dealer/Trustmark Dealer) [default Individual]: ") or "Individual")
    ind = 1 if seller_input == 'Individual' else 0
    dea = 1 if seller_input == 'Dealer' else 0
    trust = 1 if seller_input == 'Trustmark Dealer' else 0

    car_full_name = input("Car full name/model (e.g. 'Maruti Swift Dzire VDI') [optional]: ")
    brand = car_full_name.split()[0].lower() if car_full_name else "unknown"
    mileage_val = float(input("Mileage (e.g. '23.4 kmpl') [default 23.4]: ").split()[0] or 23.4)
    engine_cc_val = int(input("Engine (e.g. '1248 CC') [default 1248]: ").split()[0] or 1248)
    max_power_val = float(input("Max power (e.g. '74 bhp') [default 74]: ").split()[0] or 74)
    torque_val = float(input("Torque (e.g. '190Nm@2000rpm') [default 190]: ").split('Nm')[0] or 190)
    seats_no = int(input("Number of seats (e.g. 5) [default 5]: ") or 5)

    Age = CURRENT_YR - year
    if Age == 0: Age = 0.5
    KM_per_yr = km_driven / Age

    feature_dict = {
        'Age': Age, 'km_driven': km_driven, 'KM_per_yr': KM_per_yr, 'transmission': transmission,
        'diesel': diesel, 'petrol': petrol, 'lpg': lpg, 'cng': cng,
        '1st own': o1, '2nd own': o2, '3rd own': o3, '4+ own': o4, 'test drive own': o_test,
        'individual': ind, 'dealer': dea, 'trustmark dealer': trust,
        'Mileage_value': mileage_val, 'Engine_cc': engine_cc_val,
        'max_power': max_power_val, 'torque_value': torque_val, 'seats': seats_no
    }

    feature_cols_order = feature_col
    X_user = pd.DataFrame([feature_dict], columns=feature_cols_order).fillna(0.0)
    raw_pred = model.predict(X_user)[0]

    # Brand-based adjustment
    if brand in df['name'].str.lower().str.split().str[0].unique():
        brand_median_price = df[df['name'].str.lower().str.startswith(brand)]['selling_price'].median()
        predicted_price = 0.5*raw_pred + 0.5*brand_median_price
    else:
        predicted_price = raw_pred

    print(f"\nPredicted selling price (approx, adjusted): ₹{int(round(predicted_price))}")
    return predicted_price

# ------------------- Run Prediction -------------------
prompt_input_predict_adjusted(model, df, CURRENT_YR)
