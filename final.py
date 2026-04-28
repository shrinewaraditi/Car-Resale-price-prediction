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

def extract_brand(name):
    if pd.isna(name): return "unknown"
    return str(name).split()[0].lower()

# ---------------- LOAD DATA ----------------

df = pd.read_csv("Car details v3.csv")

# ---------------- CLEAN ----------------

df = df[df['selling_price'] > 10000]
df = df[df['selling_price'] < 10000000]

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')

df.dropna(subset=['year', 'km_driven'], inplace=True)

# ---------------- FEATURES ----------------

CURRENT_YEAR = 2025

df['Age'] = CURRENT_YEAR - df['year']
df['Age'] = df['Age'].clip(lower=0.5)

df['km_log'] = np.log1p(df['km_driven'])

# fuel
df['fuel'] = df['fuel'].fillna("Petrol")
df['is_diesel'] = (df['fuel'] == "Diesel").astype(int)
df['is_petrol'] = (df['fuel'] == "Petrol").astype(int)

# transmission
df['transmission'] = df['transmission'].fillna("Manual")
df['is_auto'] = (df['transmission'] == "Automatic").astype(int)

# owner
df['owner'] = df['owner'].fillna("First Owner")
df['owner_num'] = df['owner'].map({
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4
}).fillna(1)

# numeric extraction
df['engine'] = df['engine'].apply(extract_number)
df['max_power'] = df['max_power'].apply(extract_number)
df['mileage'] = df['mileage'].apply(extract_number)
df['torque'] = df['torque'].apply(extract_number)

# fill missing
for col in ['engine', 'max_power', 'mileage', 'torque']:
    df[col] = df[col].fillna(df[col].median())

# brand
df['brand'] = df['name'].apply(extract_brand)

# ---------------- TARGET ENCODING (simple but effective) ----------------

brand_price = df.groupby('brand')['selling_price'].mean()
df['brand_te'] = df['brand'].map(brand_price)

# normalize (IMPORTANT FIX)
df['brand_te'] = df['brand_te'] / 100000

# ---------------- FINAL FEATURES ----------------

features = [
    'Age', 'km_log',
    'engine', 'max_power', 'mileage', 'torque',
    'is_diesel', 'is_petrol',
    'is_auto', 'owner_num',
    'brand_te'
]

X = df[features]
y = np.log1p(df['selling_price'])   # log transform

# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SAMPLE WEIGHT (IMPORTANT FIX) ----------------

weights = np.expm1(y_train) / np.mean(np.expm1(y_train))

# ---------------- MODEL ----------------

model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=weights)

# ---------------- EVALUATION ----------------

y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)

print("\n===== MODEL PERFORMANCE =====")

print("R2:", r2_score(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------------- SAVE ----------------

joblib.dump(model, "car_price_xgb.pkl")
joblib.dump(brand_price, "brand_encoding.pkl")

print("\nModel saved successfully!")