import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoValue · Car Resale Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── inject custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2128;
    --accent:    #e8ff47;
    --accent2:   #47b8ff;
    --danger:    #ff6b6b;
    --text:      #f0f2f5;
    --muted:     #6b7280;
    --card-bg:   #13161d;
    --gradient:  linear-gradient(135deg, #e8ff47 0%, #47b8ff 100%);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }

.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    background: rgba(10,12,16,0.95);
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.nav-logo span { color: var(--accent); }
.nav-badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 0.8rem;
}
.hero h1 em {
    font-style: normal;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 24px; height: 2px;
    background: var(--accent);
}

.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: var(--text);
}

.result-hero {
    background: linear-gradient(135deg, #111318 0%, #13161d 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-hero::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(232,255,71,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -2px;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0.6rem;
}
.result-range {
    font-size: 0.9rem;
    color: var(--muted);
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 1.5rem;
}
.metric-chip {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-chip .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-chip .lbl {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 2px;
    letter-spacing: 0.5px;
}

.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 1.2rem;
}
.conf-high   { background: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
.conf-medium { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.conf-low    { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }

.bk-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border);
}
.bk-row:last-child { border-bottom: none; }
.bk-label { color: var(--muted); font-size: 0.88rem; }
.bk-value { font-weight: 600; font-size: 0.95rem; }
.bk-value.accent { color: var(--accent); }
.bk-value.blue   { color: var(--accent2); }

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

div.stButton > button {
    background: var(--accent) !important;
    color: #0a0c10 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    width: 100%;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88 !important; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── nav bar ────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">Auto<span>Value</span></div>
  <div class="nav-badge">India Car Resale Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ── hero ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>What's Your Car<br><em>Really Worth?</em></h1>
  <p>ML-powered resale valuation with real market depreciation curves — built for the Indian used car market.</p>
</div>
""", unsafe_allow_html=True)

# ── constants & helpers ────────────────────────────────────────
try:
    from rapidfuzz import process as fuzz_process, fuzz as fuzz_lib
    FUZZY_OK = True
except ImportError:
    FUZZY_OK = False

CURRENT_YR = 2026
LUXURY_BRANDS_SET = {'bmw','audi','mercedes-benz','jaguar','land','porsche','volvo','lexus'}
EV_KEYWORDS = {
    'ev','electric','nexon ev','zs ev','tigor ev','tiago ev',
    'e-tron','etron','id.4','id4','leaf','ioniq','kona electric',
    'mg zs ev','bz4x','nexon.ev','xev','e20'
}
SKIP_WORDS = {'suzuki','motors','rover','benz','royce','motor','renault'}

ICE_DEPRECIATION     = {0:0.00,1:0.15,2:0.28,3:0.38,4:0.45,5:0.50,6:0.53,7:0.55,8:0.57,9:0.58,10:0.60}
EV_DEPRECIATION      = {0:0.00,1:0.20,2:0.35,3:0.45,4:0.52,5:0.58,6:0.63,7:0.67,8:0.70,9:0.72,10:0.74}
LUXURY_DEPRECIATION  = {0:0.00,1:0.20,2:0.35,3:0.47,4:0.55,5:0.62,6:0.67,7:0.70,8:0.72,9:0.74,10:0.76}

def _interp(table, age):
    age_int = min(int(np.floor(age)), max(table.keys()))
    frac    = age - int(np.floor(age))
    base    = table[age_int]
    nxt     = table.get(age_int + 1, base)
    return base + frac * (nxt - base)

def extract_brand(name):
    if not name: return "unknown"
    return str(name).lower().strip().split()[0]

def extract_model(name):
    if not name: return "unknown"
    parts = str(name).lower().strip().split()
    if len(parts) < 2: return "unknown"
    if len(parts) >= 3 and parts[1] in SKIP_WORDS:
        return " ".join(parts[2:])
    return " ".join(parts[1:])

def is_electric(name, fuel):
    if any(kw in str(name).lower() for kw in EV_KEYWORDS): return True
    if str(fuel).lower() in ('electric','ev'): return True
    return False

def get_depreciation(age, car_name, fuel, brand):
    if is_electric(car_name, fuel):
        return _interp(EV_DEPRECIATION, age), "EV"
    elif brand.lower() in LUXURY_BRANDS_SET:
        return _interp(LUXURY_DEPRECIATION, age), "Luxury"
    return _interp(ICE_DEPRECIATION, age), "ICE"

def irda_depreciation(age):
    return _interp(ICE_DEPRECIATION, age)

def apply_market_correction(ml_price, purchase_price, age, km, car_name, fuel, brand):
    depr_rate, depr_label = get_depreciation(age, car_name, fuel, brand)
    market_value = purchase_price * (1 - depr_rate)
    ml_weight    = max(0.35, 1.0 - age * 0.08)
    if depr_label == "EV": ml_weight = max(0.15, ml_weight - 0.20)
    blended      = ml_weight * ml_price + (1 - ml_weight) * market_value
    expected_km  = age * 15000
    excess_km    = max(0, km - expected_km)
    km_penalty   = max(0.80, 1 - (excess_km / 10000) * 0.005)
    if km < expected_km * 0.6: km_penalty = min(1.05, km_penalty + 0.03)
    final = blended * km_penalty
    return {
        "ml_raw":       int(ml_price),
        "market_floor": int(market_value),
        "blended":      int(blended),
        "km_penalty":   round(km_penalty, 3),
        "final":        int(final),
        "depr_pct":     round(depr_rate * 100, 1),
        "depr_label":   depr_label,
        "ml_weight":    round(ml_weight, 2),
    }

@st.cache_resource
def load_model():
    if not (os.path.exists("car_price_model.pkl") and os.path.exists("encoders.pkl")):
        return None, None
    model    = joblib.load("car_price_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

xgb_model, encoders = load_model()
model_loaded = xgb_model is not None

def build_depr_df():
    rows = []
    for yr in range(0, 11):
        rows.append({"Year": yr,
                     "ICE":    round(ICE_DEPRECIATION[yr] * 100, 1),
                     "EV":     round(EV_DEPRECIATION[yr] * 100, 1),
                     "Luxury": round(LUXURY_DEPRECIATION[yr] * 100, 1)})
    return pd.DataFrame(rows).set_index("Year")

# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════
import plotly.graph_objects as go

col_form, col_right = st.columns([1.1, 1.5], gap="large")

with col_form:
    st.markdown('<div class="section-label">Valuation Input</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🚗 Vehicle Identity</div>', unsafe_allow_html=True)
    car_name = st.text_input("Car Name", placeholder="e.g. Hyundai Elite i20 Petrol Sportz BSIV")
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2019)
    with c2:
        purchase_price = st.number_input("Ex-Showroom Price (₹)", min_value=50000,
                                          max_value=20000000, value=800000, step=10000, format="%d")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📍 Usage & Condition</div>', unsafe_allow_html=True)
    km = st.number_input("KM Driven", min_value=0, max_value=500000, value=45000, step=1000)
    c3, c4 = st.columns(2)
    with c3:
        fuel = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","LPG","Electric"])
    with c4:
        trans = st.selectbox("Transmission", ["Manual","Automatic"])
    owner  = st.selectbox("Ownership", ["First Owner","Second Owner","Third Owner",
                                         "Fourth & Above Owner","Test Drive Owner"])
    seller = st.selectbox("Seller Type", ["Individual","Dealer","Trustmark Dealer"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">⚙️ Technical Specs</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.0, step=0.5)
        engine  = st.number_input("Engine CC (0 for EV)", min_value=0, max_value=6000, value=1197)
    with c6:
        power  = st.number_input("Max Power (bhp)", min_value=0.0, max_value=500.0, value=82.0, step=1.0)
        torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=1000.0, value=113.0, step=1.0)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("⚡ Estimate Resale Value", type="primary")

with col_right:
    # ── Always-visible depreciation chart ──
    st.markdown('<div class="section-label">Market Depreciation Curves</div>', unsafe_allow_html=True)
    depr_df = build_depr_df()

    fig_depr = go.Figure()
    curve_colors = {"ICE": "#e8ff47", "EV": "#47b8ff", "Luxury": "#ff6b6b"}
    fill_map     = {"ICE": "rgba(232,255,71,0.06)", "EV": "rgba(71,184,255,0.06)", "Luxury": "rgba(255,107,107,0.06)"}
    for name_c, color in curve_colors.items():
        fig_depr.add_trace(go.Scatter(
            x=depr_df.index, y=depr_df[name_c],
            mode='lines+markers', name=name_c,
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            fill='tozeroy', fillcolor=fill_map[name_c],
            hovertemplate=f'{name_c} — Year %{{x}}: <b>%{{y}}% depreciation</b><extra></extra>',
        ))
    fig_depr.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#6b7280'),
        legend=dict(orientation='h', y=-0.2, bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f2f5', size=11)),
        margin=dict(l=0, r=0, t=10, b=50),
        xaxis=dict(title='Car Age (years)', gridcolor='#1e2128',
                   tickfont=dict(color='#6b7280'), title_font=dict(color='#6b7280'), zeroline=False),
        yaxis=dict(title='% Value Lost', gridcolor='#1e2128',
                   tickfont=dict(color='#6b7280'), title_font=dict(color='#6b7280'), zeroline=False),
        hovermode='x unified', height=240,
    )
    st.plotly_chart(fig_depr, use_container_width=True)

    # ── Results ──────────────────────────────────────────────
    if predict_clicked:
        if not car_name.strip():
            st.error("Please enter a car name to get an estimate.")
            st.stop()

        age       = float(CURRENT_YR - year)
        if age <= 0: age = 0.5
        brand_key = extract_brand(car_name)
        model_key = extract_model(car_name)
        bm_key    = brand_key + "_" + model_key
        ev_flag   = is_electric(car_name, fuel.upper())

        if model_loaded:
            enc              = encoders
            brand_map        = enc["brand_map"]
            model_map        = enc["model_map"]
            bm_map           = enc["bm_map"]
            global_mean      = enc["global_mean"]
            feature_col      = enc["feature_cols"]
            brand_median_map = enc.get("brand_median_map", {})

            final_brand_te = brand_map.get(brand_key, global_mean)

            if bm_key in bm_map:
                final_bm_te    = bm_map[bm_key]
                final_model_te = model_map.get(model_key, brand_median_map.get(brand_key, global_mean))
                match_label    = "Exact Match"
                conf_cls       = "conf-high"; conf_icon = "🟢"
            else:
                candidates = {k: k.split('_', 1)[-1] for k in bm_map if brand_key in k}
                best_key   = None
                if candidates:
                    if FUZZY_OK:
                        res = fuzz_process.extractOne(model_key, candidates, scorer=fuzz_lib.token_set_ratio)
                        if res and res[1] >= 55:
                            best_key = res[2]
                    else:
                        model_words = set(model_key.split())
                        scored = []
                        for k in candidates:
                            overlap = len(model_words & set(k.split())) / max(len(model_words), 1)
                            scored.append((k, overlap))
                        scored.sort(key=lambda x: -x[1])
                        if scored and scored[0][1] > 0:
                            best_key = scored[0][0]

                if best_key:
                    final_bm_te    = bm_map[best_key]
                    final_model_te = model_map.get(best_key.split('_', 1)[-1],
                                     brand_median_map.get(brand_key, global_mean))
                    match_label = "Fuzzy Match"
                    conf_cls    = "conf-medium"; conf_icon = "🟡"
                else:
                    final_bm_te    = brand_median_map.get(brand_key, global_mean)
                    final_model_te = brand_median_map.get(brand_key, global_mean)
                    match_label = "Brand Median"
                    conf_cls    = "conf-low"; conf_icon = "🔴"

            engine_cc = engine if engine > 0 else 1
            fuel_up   = fuel.upper()

            row = {
                'Age': age, 'Age_squared': age**2,
                'km_driven': km, 'km_driven_log': np.log1p(km),
                'KM_per_yr': km / age,
                'depreciation_factor': 1 - irda_depreciation(age),
                'age_km_interaction': age * km,
                'transmission': 0 if trans.lower().startswith('m') else 1,
                'diesel': int(fuel_up == 'DIESEL'), 'petrol': int(fuel_up == 'PETROL'),
                'lpg':    int(fuel_up == 'LPG'),    'cng':    int(fuel_up == 'CNG'),
                '1st own':        int(owner == 'First Owner'),
                '2nd own':        int(owner == 'Second Owner'),
                '3rd own':        int(owner == 'Third Owner'),
                '4+ own':         int(owner == 'Fourth & Above Owner'),
                'test drive own': int(owner == 'Test Drive Owner'),
                'individual':       int(seller == 'Individual'),
                'dealer':           int(seller == 'Dealer'),
                'trustmark dealer': int(seller == 'Trustmark Dealer'),
                'Mileage_value': mileage, 'Engine_cc': engine_cc,
                'max_power': power, 'torque_value': torque, 'seats': int(seats),
                'power_to_weight': power / (engine_cc / 1000),
                'is_luxury': int(brand_key in LUXURY_BRANDS_SET),
                'brand_te':  final_brand_te,
                'model_te':  final_model_te,
                'bm_te':     final_bm_te,
            }
            X_user  = pd.DataFrame([row])[feature_col]
            ml_pred = np.expm1(xgb_model.predict(X_user)[0])
            result  = apply_market_correction(
                ml_pred, purchase_price, age, km, car_name, fuel_up, brand_key
            )

        else:
            # Demo / market-curve-only mode (no model files present)
            depr_rate, depr_label = get_depreciation(age, car_name, fuel.upper(), brand_key)
            market_val = purchase_price * (1 - depr_rate)
            expected_km = age * 15000
            excess_km   = max(0, km - expected_km)
            km_penalty  = max(0.80, 1 - (excess_km / 10000) * 0.005)
            if km < expected_km * 0.6: km_penalty = min(1.05, km_penalty + 0.03)
            result = {
                "ml_raw":       int(market_val * 1.07),
                "market_floor": int(market_val),
                "blended":      int(market_val * 1.03),
                "km_penalty":   round(km_penalty, 3),
                "final":        int(market_val * km_penalty),
                "depr_pct":     round(depr_rate * 100, 1),
                "depr_label":   depr_label,
                "ml_weight":    0.50,
            }
            match_label = "Market Curve Only"
            conf_cls    = "conf-low"; conf_icon = "🔴"

        final   = result["final"]
        low     = int(final * 0.92)
        high    = int(final * 1.08)
        val_pct = round((final / purchase_price) * 100, 1)
        depr_icon = {'EV': '⚡', 'Luxury': '💎', 'ICE': '🔥'}.get(result['depr_label'], '🔥')

        # ── Hero result card ──
        st.markdown(f"""
        <div class="result-hero">
          <div class="result-label">Estimated Resale Value</div>
          <div class="result-price">₹{final:,}</div>
          <div class="result-range">Realistic Range &nbsp;·&nbsp; ₹{low:,} – ₹{high:,} &nbsp;(±8%)</div>
          <span class="conf-badge {conf_cls}">{conf_icon} {match_label}</span>
          <div class="metric-row">
            <div class="metric-chip">
              <div class="val">{result['depr_pct']}%</div>
              <div class="lbl">{depr_icon} {result['depr_label']} Depr.</div>
            </div>
            <div class="metric-chip">
              <div class="val">{val_pct}%</div>
              <div class="lbl">Value Retained</div>
            </div>
            <div class="metric-chip">
              <div class="val">{result['km_penalty']:.1%}</div>
              <div class="lbl">KM Factor</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ──────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📊 Price Breakdown", "📉 Depreciation Track", "📈 Comparison"])

        with tab1:
            st.markdown(f"""
            <div class="card">
              <div class="bk-row">
                <span class="bk-label">Original Price</span>
                <span class="bk-value">₹{purchase_price:,}</span>
              </div>
              <div class="bk-row">
                <span class="bk-label">🤖 ML Raw Estimate</span>
                <span class="bk-value blue">₹{result['ml_raw']:,}</span>
              </div>
              <div class="bk-row">
                <span class="bk-label">{depr_icon} Market Floor ({result['depr_label']})</span>
                <span class="bk-value">₹{result['market_floor']:,} &nbsp;<small style="color:#6b7280">({result['depr_pct']}% depr.)</small></span>
              </div>
              <div class="bk-row">
                <span class="bk-label">🔀 Blended ({result['ml_weight']:.0%} ML + {1-result['ml_weight']:.0%} market)</span>
                <span class="bk-value">₹{result['blended']:,}</span>
              </div>
              <div class="bk-row">
                <span class="bk-label">🛣️ KM Adjustment Factor</span>
                <span class="bk-value">{result['km_penalty']:.1%}</span>
              </div>
              <div class="bk-row">
                <span class="bk-label" style="font-weight:700;color:#f0f2f5">💰 Final Estimate</span>
                <span class="bk-value accent" style="font-size:1.15rem">₹{final:,}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            ages_range   = np.linspace(0, 10, 81)
            dep_table    = {'ICE': ICE_DEPRECIATION, 'EV': EV_DEPRECIATION, 'Luxury': LUXURY_DEPRECIATION}[result['depr_label']]
            vals         = [purchase_price * (1 - _interp(dep_table, a)) for a in ages_range]
            car_val_now  = result['market_floor']

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=ages_range, y=vals, mode='lines', name='Value Curve',
                line=dict(color='#47b8ff', width=2.5),
                fill='tozeroy', fillcolor='rgba(71,184,255,0.06)',
            ))
            fig2.add_trace(go.Scatter(
                x=[age], y=[car_val_now], mode='markers', name='Your Car Now',
                marker=dict(size=14, color='#e8ff47', symbol='circle',
                            line=dict(color='#0a0c10', width=2)),
                hovertemplate=f'Age {age:.1f}y — <b>₹{car_val_now:,}</b><extra></extra>',
            ))
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#6b7280'),
                legend=dict(orientation='h', y=-0.25, bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f2f5', size=11)),
                margin=dict(l=0, r=0, t=10, b=50),
                xaxis=dict(title='Age (years)', gridcolor='#1e2128',
                           tickfont=dict(color='#6b7280'), title_font=dict(color='#6b7280'), zeroline=False),
                yaxis=dict(title='Market Value (₹)', gridcolor='#1e2128',
                           tickfont=dict(color='#6b7280'), title_font=dict(color='#6b7280'),
                           zeroline=False, tickformat=',.0f'),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            labels     = ["Original<br>Price", "ML Raw<br>Estimate", "Market<br>Floor", "Final<br>Estimate"]
            values     = [purchase_price, result['ml_raw'], result['market_floor'], final]
            bar_colors = ["#6b7280", "#47b8ff", "#fbbf24", "#e8ff47"]

            fig3 = go.Figure(go.Bar(
                x=labels, y=values,
                marker_color=bar_colors,
                marker_line_width=0,
                text=[f"₹{v:,}" for v in values],
                textposition='outside',
                textfont=dict(color='#f0f2f5', size=11),
            ))
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#6b7280'),
                margin=dict(l=0, r=0, t=40, b=20),
                xaxis=dict(gridcolor='#1e2128', tickfont=dict(color='#f0f2f5', size=12), zeroline=False),
                yaxis=dict(gridcolor='#1e2128', tickfont=dict(color='#6b7280'),
                           zeroline=False, tickformat=',.0f'),
                showlegend=False, height=290,
            )
            st.plotly_chart(fig3, use_container_width=True)

        if ev_flag:
            st.info("⚡ **EV detected** — EV depreciation curve applied. Battery anxiety + limited resale demand cause faster depreciation in India.")

        # if not model_loaded:
        #     st.warning("⚠️ **Demo mode** — `car_price_model.pkl` and `encoders.pkl` not found. Showing market-curve-only estimate. Place your trained model files alongside this script for full ML predictions.")

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3.5rem 1.5rem; border-style:dashed;">
          <div style="font-size:3rem; margin-bottom:1rem">🚗</div>
          <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-bottom:0.6rem">
            Ready for Valuation
          </div>
          <div style="color:#6b7280; font-size:0.9rem; line-height:1.7">
            Fill in the form on the left and click<br>
            <strong style="color:#e8ff47">⚡ Estimate Resale Value</strong><br>
            for an instant ML-powered result.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── footer ──────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#3a3f4a; font-size:0.78rem; padding:1rem 0 2rem;">
  AutoValue &nbsp;·&nbsp; XGBoost + Market Depreciation Curves &nbsp;·&nbsp;
  Trained on CarDekho India dataset &nbsp;·&nbsp; 🇮🇳
</div>
""", unsafe_allow_html=True)