"""
Parkinson's Disease Prediction Web Application
================================================
A multi-tab Streamlit app supporting two prediction pathways:
  Tab 1 - Lifestyle & Clinical Assessment  (Random Forest model)
  Tab 2 - Vocal Acoustic Assessment        (Ensemble: RF + SVM + XGBoost)

All .pkl files are resolved relative to this script's location so the app
works in any environment without hardcoded absolute paths.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config  (must be the very first Streamlit call)
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - medical teal/navy theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Medical navy background */
    .stApp { background: linear-gradient(160deg, #e8f4f8 0%, #ddeef6 50%, #cfe6f0 100%); }

    .hero-banner {
        background: linear-gradient(120deg, #0a3d5c, #0e5a84);
        border: 1px solid #1a7fb5;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(10,61,92,0.35);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7ee8ff, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #b8dded;
        margin: 0;
        line-height: 1.6;
    }

    .result-healthy {
        background: linear-gradient(135deg, #e6f9f0, #c8f0de);
        border: 2px solid #16a34a;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-parkinsons {
        background: linear-gradient(135deg, #fff1f1, #ffe0e0);
        border: 2px solid #dc2626;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-title-healthy  { font-size: 1.5rem; font-weight: 700; color: #15803d; }
    .result-title-disease  { font-size: 1.5rem; font-weight: 700; color: #b91c1c; }
    .result-prob           { font-size: 0.95rem; color: #374151; margin-top: 0.4rem; }

    /* Sidebar - deep medical navy */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07293f 0%, #0a3d5c 100%) !important;
        border-right: 1px solid #1a7fb5;
    }
    [data-testid="stSidebar"] * { color: #c8e8f5 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 10px 10px 0 0;
        border: 1px solid #a8d4e8;
        color: #0a3d5c !important;
        font-weight: 500;
        padding: 0.6rem 1.4rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0a3d5c, #0e6da8) !important;
        border-color: transparent !important;
        color: #ffffff !important;
    }

    /* Predict button - teal medical */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0e6da8, #0891b2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(14,109,168,0.4);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14,109,168,0.6);
    }

    /* Input/widget accent */
    .stSlider > div > div > div > div { background: #0891b2 !important; }
    hr { border-color: #a8d4e8 !important; }
    .stAlert { border-radius: 10px !important; }

    /* Form container */
    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #bee3f8;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(10,61,92,0.08);
    }

    /* Labels and text on white background */
    label, .stSelectbox label, .stNumberInput label { color: #0a3d5c !important; font-weight: 500; }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0e6da8;
        margin-bottom: 0.6rem;
    }

    /* Main heading text */
    h3, h4, h5 { color: #0a3d5c !important; }
    p, .stMarkdown p { color: #1e3a4f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helpers - resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(BASE_DIR, filename)


# Cached model loaders
@st.cache_resource(show_spinner="Loading Lifestyle model...")
def load_lifestyle_assets():
    errors = []
    for fname in ("parkinsons_model.pkl", "scaler.pkl", "features.pkl"):
        if not os.path.exists(_path(fname)):
            errors.append(fname)
    if errors:
        return None, None, None, errors
    model    = joblib.load(_path("parkinsons_model.pkl"))
    scaler   = joblib.load(_path("scaler.pkl"))
    features = joblib.load(_path("features.pkl"))
    return model, scaler, features, []


@st.cache_resource(show_spinner="Loading Voice model...")
def load_voice_assets():
    """
    Real pkl contents (names in repo are misleading):
      voice_scaler.pkl        -> SimpleImputer   (28 raw features)
      voice_rfe.pkl           -> RFE selector    (selects 11 of 28)
      voice_features.pkl      -> StandardScaler  (scales 11 selected features)
      parkinsons_voice_model.pkl -> RandomForestClassifier (expects 11 features)
    Returns: (model, imputer, rfe, scaler, all_features, selected_features, errors)
    """
    errors = []
    for fname in ("parkinsons_voice_model.pkl", "voice_scaler.pkl",
                  "voice_features.pkl", "voice_rfe.pkl"):
        if not os.path.exists(_path(fname)):
            errors.append(fname)
    if errors:
        return None, None, None, None, [], [], errors

    model   = joblib.load(_path("parkinsons_voice_model.pkl"))
    imputer = joblib.load(_path("voice_scaler.pkl"))    # SimpleImputer
    rfe     = joblib.load(_path("voice_rfe.pkl"))       # RFE selector
    scaler  = joblib.load(_path("voice_features.pkl"))  # StandardScaler

    # Derive ordered feature lists from the imputer's saved metadata
    all_features      = list(imputer.feature_names_in_)
    selected_features = [f for f, s in zip(all_features, rfe.support_) if s]

    return model, imputer, rfe, scaler, all_features, selected_features, []


# Result renderer
def render_result(prediction, probability):
    prob_pct = probability * 100
    if prediction == 0:
        st.markdown(
            f"""<div class="result-healthy">
                <div class="result-title-healthy">✅ No Parkinson's Detected</div>
                <div class="result-prob">Confidence: <strong>{prob_pct:.1f}%</strong> probability of being healthy</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="result-parkinsons">
                <div class="result-title-disease">⚠️ Parkinson's Disease Detected</div>
                <div class="result-prob">
                    Confidence: <strong>{prob_pct:.1f}%</strong> probability of Parkinson's Disease<br>
                    <em>Please consult a qualified neurologist for a clinical evaluation.</em>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">🧠 Parkinson's Disease Prediction</p>
        <p class="hero-subtitle">
            An AI-powered screening tool using two independent prediction pathways —
            <strong>Lifestyle &amp; Clinical</strong> (Random Forest) and
            <strong>Vocal Acoustic</strong> (Ensemble: RF + SVM + XGBoost).<br>
            <em>This tool is intended for informational purposes only and does not replace professional medical advice.</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Instructions")
    st.markdown("""
        1. Choose an assessment tab above.
        2. Fill in all the required fields.
        3. Click **Predict** to see the result.
        4. Review the confidence score and consult a doctor if needed.
    """)
    st.markdown("---")
    st.markdown("### ℹ️ About the Models")
    st.markdown("""
        **Tab 1 – Lifestyle & Clinical**
        Uses a *Random Forest* classifier trained on demographic,
        motor, cognitive, and lifestyle features.

        **Tab 2 – Vocal Acoustic**
        Uses a voting *Ensemble* of Random Forest, SVM, and XGBoost
        trained on MDVP / Jitter / Shimmer / nonlinear vocal biomarkers.
    """)
    st.markdown("---")


# Main tabs
tab1, tab2 = st.tabs(["🏥  Lifestyle & Clinical Assessment", "🎙️  Vocal Acoustic Assessment"])

# TAB 1 - Lifestyle & Clinical Assessment
with tab1:
    st.markdown("### 🏥 Lifestyle & Clinical Assessment")
    st.markdown(
        "Enter the patient's demographic, clinical, and lifestyle information below. "
        "All continuous values will be automatically scaled before prediction."
    )

    lf_model, lf_scaler, lf_features, lf_errors = load_lifestyle_assets()

    if lf_errors:
        for fname in lf_errors:
            st.error(f"❌ Missing required file: **{fname}**  \nPlease ensure it is in the same directory as `app.py`.")
        st.stop()

    CONTINUOUS_FEATURES = ["UPDRS", "MoCA", "FunctionalAssessment", "Age", "BMI", "AlcoholConsumption"]
    BINARY_FEATURES     = ["Tremor", "Rigidity", "Bradykinesia", "PosturalInstability", "Depression", "Diabetes"]

    with st.form("lifestyle_form"):
        st.markdown('<div class="section-label">📊 Continuous Clinical Measurements</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age  = st.number_input("Age (years)",       min_value=18,   max_value=110,   value=60,   step=1)
            bmi  = st.number_input("BMI (kg/m²)",       min_value=10.0, max_value=60.0,  value=25.0, step=0.1, format="%.1f")
        with col2:
            updrs = st.number_input("UPDRS Score",       min_value=0.0,  max_value=200.0, value=30.0, step=0.1, format="%.1f",
                                    help="Unified Parkinson's Disease Rating Scale (0 = no impairment)")
            moca  = st.number_input("MoCA Score",        min_value=0.0,  max_value=30.0,  value=24.0, step=0.1, format="%.1f",
                                    help="Montreal Cognitive Assessment (0–30; ≥26 is normal)")
        with col3:
            func_assessment = st.number_input("Functional Assessment", min_value=0.0,  max_value=100.0, value=70.0, step=0.1, format="%.1f",
                                              help="Composite functional ability score (0–100)")
            alcohol         = st.number_input("Alcohol Consumption (units/week)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, format="%.1f")

        st.markdown("---")
        st.markdown('<div class="section-label">🔘 Symptoms & Co-morbidities</div>', unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        yes_no = {"No": 0, "Yes": 1}

        with col4:
            tremor   = yes_no[st.selectbox("Tremor",               options=["No", "Yes"], help="Resting tremor present?")]
            rigidity = yes_no[st.selectbox("Rigidity",             options=["No", "Yes"], help="Muscle rigidity present?")]
        with col5:
            brady    = yes_no[st.selectbox("Bradykinesia",         options=["No", "Yes"], help="Slowness of movement?")]
            postural = yes_no[st.selectbox("Postural Instability", options=["No", "Yes"], help="Balance impairment?")]
        with col6:
            depression = yes_no[st.selectbox("Depression",  options=["No", "Yes"], help="Diagnosed with depression?")]
            diabetes   = yes_no[st.selectbox("Diabetes",    options=["No", "Yes"], help="Diagnosed with diabetes?")]

        st.markdown("---")
        submitted_lf = st.form_submit_button("🔍 Predict – Lifestyle Assessment")

    if submitted_lf:
        try:
            user_inputs = {
                "Age":                 age,
                "BMI":                 bmi,
                "UPDRS":               updrs,
                "MoCA":                moca,
                "FunctionalAssessment":func_assessment,
                "AlcoholConsumption":  alcohol,
                "Tremor":              tremor,
                "Rigidity":            rigidity,
                "Bradykinesia":        brady,
                "PosturalInstability": postural,
                "Depression":          depression,
                "Diabetes":            diabetes,
            }

            input_df = pd.DataFrame([user_inputs])
            cols_to_scale = [c for c in CONTINUOUS_FEATURES if c in input_df.columns]
            input_df[cols_to_scale] = lf_scaler.transform(input_df[cols_to_scale])
            input_df = input_df[lf_features]

            prediction  = int(lf_model.predict(input_df)[0])
            proba       = lf_model.predict_proba(input_df)[0]
            probability = float(proba[prediction])

            st.markdown("---")
            st.markdown("#### Prediction Result")
            render_result(prediction, probability)

        except KeyError as e:
            st.error(f"❌ **Column mismatch**: Feature **{e}** not found. Check that features.pkl names match input column names.")
        except Exception as e:
            st.error(f"❌ Unexpected error: `{e}`")


# TAB 2 - Vocal Acoustic Assessment
with tab2:
    st.markdown("### 🎙️ Vocal Acoustic Assessment")
    st.markdown(
        "Enter the patient's vocal biomarker measurements. "
        "The pipeline automatically imputes missing values, applies RFE feature selection, scales, then predicts."
    )

    v_model, v_imputer, v_rfe, v_scaler, v_all_features, v_sel_features, v_errors = load_voice_assets()

    if v_errors:
        for fname in v_errors:
            st.error(f"Missing required file: **{fname}**  \nPlease ensure it is in the same directory as `app.py`.")
        st.stop()

    LABEL_MAP = {
        "MDVP:Fo(Hz)":      "MDVP: Fundamental Frequency - Fo (Hz)",
        "MDVP:Fhi(Hz)":     "MDVP: Maximum Frequency - Fhi (Hz)",
        "MDVP:Flo(Hz)":     "MDVP: Minimum Frequency - Flo (Hz)",
        "MDVP:Jitter(%)":   "MDVP: Jitter (%)",
        "MDVP:Jitter(Abs)": "MDVP: Jitter Absolute (s)",
        "MDVP:RAP":         "MDVP: RAP (Relative Avg Perturbation)",
        "MDVP:PPQ":         "MDVP: PPQ (5-pt Period Perturbation)",
        "Jitter:DDP":       "Jitter: DDP",
        "MDVP:Shimmer":     "MDVP: Shimmer",
        "MDVP:Shimmer(dB)": "MDVP: Shimmer (dB)",
        "Shimmer:APQ3":     "Shimmer: APQ3",
        "Shimmer:APQ5":     "Shimmer: APQ5",
        "MDVP:APQ":         "MDVP: APQ (Amplitude Perturbation)",
        "Shimmer:DDA":      "Shimmer: DDA",
        "NHR":              "NHR (Noise-to-Harmonics Ratio)",
        "HNR":              "HNR (Harmonics-to-Noise Ratio) dB",
        "RPDE":             "RPDE (Recurrence Period Density Entropy)",
        "DFA":              "DFA (Detrended Fluctuation Analysis)",
        "spread1":          "Spread1 (Nonlinear Measure)",
        "spread2":          "Spread2 (Nonlinear Measure)",
        "D2":               "D2 (Correlation Dimension)",
        "PPE":              "PPE (Pitch Period Entropy)",
        "jitter_shimmer":   "Jitter x Shimmer (engineered)",
        "spread_combined":  "Spread Combined (engineered)",
        "noise_ratio":      "Noise Ratio (engineered)",
        "complexity":       "Complexity (engineered)",
        "jitter_mean":      "Jitter Mean (engineered)",
        "shimmer_mean":     "Shimmer Mean (engineered)",
    }

    DEFAULTS = {
        "MDVP:Fo(Hz)": 154.23, "MDVP:Fhi(Hz)": 197.11, "MDVP:Flo(Hz)": 116.32,
        "MDVP:Jitter(%)": 0.00622, "MDVP:Jitter(Abs)": 0.00004,
        "MDVP:RAP": 0.00301, "MDVP:PPQ": 0.00317, "Jitter:DDP": 0.00902,
        "MDVP:Shimmer": 0.02971, "MDVP:Shimmer(dB)": 0.282,
        "Shimmer:APQ3": 0.01489, "Shimmer:APQ5": 0.01804,
        "MDVP:APQ": 0.02488, "Shimmer:DDA": 0.04466,
        "NHR": 0.01149, "HNR": 21.034,
        "RPDE": 0.499, "DFA": 0.718,
        "spread1": -5.684, "spread2": 0.227, "D2": 2.301, "PPE": 0.284,
        "jitter_shimmer": 0.000185, "spread_combined": -5.457,
        "noise_ratio": 0.011, "complexity": 2.301,
        "jitter_mean": 0.00478, "shimmer_mean": 0.022,
    }

    st.info(
        f"**{len(v_all_features)} raw features** collected "
        f"-> imputed -> RFE selects **{len(v_sel_features)}** "
        f"-> scaled -> predicted.  \n"
        f"RFE-selected features: `{'`, `'.join(v_sel_features)}`"
    )

    with st.form("voice_form"):
        n_cols = 3
        cols   = st.columns(n_cols)
        voice_inputs = {}

        for idx, feature in enumerate(v_all_features):
            label   = LABEL_MAP.get(feature, feature)
            default = float(DEFAULTS.get(feature, 0.0))
            col     = cols[idx % n_cols]
            with col:
                voice_inputs[feature] = st.number_input(
                    label=label,
                    value=default,
                    format="%.5f",
                    key=f"voice_{feature}",
                    help=f"Raw feature key: {feature}",
                )

        st.markdown("---")
        submitted_v = st.form_submit_button("🔍 Predict – Vocal Assessment")

    if submitted_v:
        try:
            # Step 1: Build DataFrame with all 28 raw features in correct order
            input_df = pd.DataFrame([voice_inputs])[v_all_features]

            # Step 2: Impute (SimpleImputer handles NaN / missing values)
            input_arr = v_imputer.transform(input_df)
            input_df  = pd.DataFrame(input_arr, columns=v_all_features)

            # Step 3: RFE feature selection - keep only the 11 selected columns
            input_df = input_df[v_sel_features]

            # Step 4: Scale the 11 selected features
            scaled_arr = v_scaler.transform(input_df)
            input_df   = pd.DataFrame(scaled_arr, columns=v_sel_features)

            # Step 5: Predict
            prediction  = int(v_model.predict(input_df)[0])
            proba       = v_model.predict_proba(input_df)[0]
            probability = float(proba[prediction])

            st.markdown("---")
            st.markdown("#### Prediction Result")
            render_result(prediction, probability)

        except KeyError as e:
            st.error(f"Column mismatch: Feature {e} not found in the pipeline. "
                     "Check that feature names match what the model was trained on.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


