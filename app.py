import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="CardioPredict - Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme toggle setup
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# CSS variables based on theme
bg_color = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0d0d11" if IS_DARK else "#f9fafb"
card_color = "#0d0d11" if IS_DARK else "#ffffff"
card_hover = "#14141a" if IS_DARK else "#f4f4f5"
border_color = "#1f1f26" if IS_DARK else "#e4e4e7"
border_subtle = "#171720" if IS_DARK else "#f0f0f2"
text_color = "#fafafa" if IS_DARK else "#09090b"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
green_color = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red_color = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
amber_color = "#f59e0b" if IS_DARK else "#d97706"
amber_muted = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"
shadow_val = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

css_code = f"""
<style>
:root {{
    --bg: {bg_color};
    --bg-subtle: {bg_subtle};
    --card: {card_color};
    --card-hover: {card_hover};
    --border: {border_color};
    --border-subtle: {border_subtle};
    --text: {text_color};
    --text-muted: #71717a;
    --text-dim: {text_dim};
    --accent: #3b82f6;
    --accent-muted: #2563eb;
    --green: {green_color};
    --green-muted: {green_muted};
    --red: {red_color};
    --red-muted: {red_muted};
    --amber: {amber_color};
    --amber-muted: {amber_muted};
    --shadow: {shadow_val};
    --radius: 10px;
}}

/* Hide default Streamlit decoration */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1360px !important;
}}

/* Custom styled inputs/cards */
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}}
.metric-label {{
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
}}
.metric-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
}}

.chart-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}
.chart-title {{
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
}}
.chart-subtitle {{
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
}}

/* Custom styling for tabs */
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
    transition: all 0.2s ease !important;
}}
button[data-baseweb="tab"]:hover {{
    color: var(--text) !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}
[data-baseweb="tab-list"] {{
    gap: 6px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 1.5rem !important;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 800;
    font-size: 1.6rem;
    color: var(--text);
    letter-spacing: -0.04em;
}}
.brand span {{
    color: var(--red);
}}

/* Custom alerts */
.alert-box {{
    padding: 1.25rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin-top: 1.25rem;
    margin-bottom: 1.25rem;
}}
.alert-success {{
    background: var(--green-muted);
    border-color: var(--green);
    color: var(--green);
}}
.alert-danger {{
    background: var(--red-muted);
    border-color: var(--red);
    color: var(--red);
}}

/* Column spacing */
[data-testid="stHorizontalBlock"] {{
    gap: 1.5rem !important;
}}
</style>
"""
st.markdown(css_code, unsafe_allow_html=True)

# Helper functions for UI components
def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# Plotly styling configuration
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        zerolinecolor="rgba(0,0,0,0.06)" if not IS_DARK else "rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

# Header Section
head_left, head_right = st.columns([8, 2])
with head_left:
    st.markdown('<div class="brand">❤️ Cardio<span>Predict</span></div>', unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, key="theme_toggle", use_container_width=True)

# Load resources
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("models/heart_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data
def load_dataset():
    values = pd.read_csv("data/values.csv")
    labels = pd.read_csv("data/labels.csv")
    return pd.merge(values, labels, on="patient_id")

model, scaler = load_model_and_scaler()

# Tabs Navigation
tab1, tab2, tab3 = st.tabs(["🏥 Diagnosis & Prediction", "📊 Patient EDA & Insights", "ℹ️ About the Project"])

with tab1:
    st.write("Fill in the patient's medical details below to predict the probability of heart disease presence.")
    
    # Columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Vitals")
        age = st.slider("Age (years)", 18, 100, 54)
        sex = st.selectbox("Biological Sex", ["Female", "Male"], index=1)
        bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 560, 240)
        max_hr = st.slider("Maximum Heart Rate Achieved (bpm)", 60, 220, 150)
        
    with col2:
        st.subheader("Cardiovascular Signs")
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["1: Typical Angina", "2: Atypical Angina", "3: Non-anginal Pain", "4: Asymptomatic"],
            index=3
        )
        thal = st.selectbox(
            "Thal (Thalassemia blood test)",
            ["Normal", "Fixed Defect", "Reversible Defect"],
            index=0
        )
        angina = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"],
            index=0
        )

    # Advanced Clinical Inputs (Expander)
    st.write("")
    with st.expander("🛠️ Advanced Clinical Parameters (Optional/Set Defaults)"):
        st.write("These clinical parameters default to standard baseline values if unknown.")
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            slope = st.selectbox(
                "Slope of the Peak Exercise ST Segment",
                ["1: Upsloping", "2: Flat", "3: Downsloping"],
                index=0
            )
            num_vessels = st.slider("Number of Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, 0)
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
                index=0
            )
            
        with adv_col2:
            rest_ekg = st.selectbox(
                "Resting Electrocardiographic Results",
                ["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"],
                index=0
            )
            oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest (Oldpeak)", 0.0, 6.2, 0.5, step=0.1)

    # Predict Trigger
    st.write("")
    if st.button("Run Diagnostic Analysis", type="primary", use_container_width=True):
        # Format Inputs
        sex_val = 1 if sex == "Male" else 0
        angina_val = 1 if angina == "Yes" else 0
        fbs_val = 1 if "Yes" in fbs else 0
        
        # Extract numerical category index
        chest_pain_val = int(chest_pain.split(":")[0])
        slope_val = int(slope.split(":")[0])
        rest_ekg_val = int(rest_ekg.split(":")[0])
        
        # One-hot encode thal
        thal_fixed = 1 if thal == "Fixed Defect" else 0
        thal_normal = 1 if thal == "Normal" else 0
        thal_reversible = 1 if thal == "Reversible Defect" else 0
        
        # Construct DataFrame in exactly the same order as training features
        # Training feature columns order:
        # ['slope_of_peak_exercise_st_segment', 'resting_blood_pressure', 'chest_pain_type', 'num_major_vessels', 
        #  'fasting_blood_sugar_gt_120_mg_per_dl', 'resting_ekg_results', 'serum_cholesterol_mg_per_dl', 
        #  'oldpeak_eq_st_depression', 'sex', 'age', 'max_heart_rate_achieved', 'exercise_induced_angina', 
        #  'thal_fixed_defect', 'thal_normal', 'thal_reversible_defect']
        sample = pd.DataFrame([[
            slope_val,
            bp,
            chest_pain_val,
            num_vessels,
            fbs_val,
            rest_ekg_val,
            chol,
            np.log1p(oldpeak),  # Apply log1p transformation to oldpeak (matches training preprocessing)
            sex_val,
            age,
            max_hr,
            angina_val,
            thal_fixed,
            thal_normal,
            thal_reversible
        ]], columns=[
            'slope_of_peak_exercise_st_segment',
            'resting_blood_pressure',
            'chest_pain_type',
            'num_major_vessels',
            'fasting_blood_sugar_gt_120_mg_per_dl',
            'resting_ekg_results',
            'serum_cholesterol_mg_per_dl',
            'oldpeak_eq_st_depression',
            'sex',
            'age',
            'max_heart_rate_achieved',
            'exercise_induced_angina',
            'thal_fixed_defect',
            'thal_normal',
            'thal_reversible_defect'
        ])
        
        # Scale continuous features
        cols_to_scale = [
            'age',
            'resting_blood_pressure',
            'serum_cholesterol_mg_per_dl',
            'max_heart_rate_achieved',
            'oldpeak_eq_st_depression'
        ]
        sample[cols_to_scale] = scaler.transform(sample[cols_to_scale])
        
        # Predict Class and Probability
        prediction = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0]
        prob_heart_disease = prob[1]
        
        # Custom HTML Result Render
        if prediction == 1:
            st.markdown(f"""
            <div class="alert-box alert-danger">
                <h3 style="margin-top:0;">⚠️ High Risk Detected</h3>
                <p>The Support Vector Machine (SVM) model predicts that <strong>Heart Disease is present</strong>.</p>
                <p style="font-size:1.15rem;"><strong>Confidence Score: {prob_heart_disease * 100:.1f}%</strong></p>
                <hr style="border-color: var(--red); opacity: 0.2;" />
                <p style="font-size:0.85rem; margin-bottom:0;"><em>Disclaimer: This assessment is based on a statistical machine learning model. It is for educational/screening purposes and does not substitute for a professional clinical diagnosis. Please consult a cardiologist or medical practitioner.</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box alert-success">
                <h3 style="margin-top:0;">✅ Low Risk Detected</h3>
                <p>The Support Vector Machine (SVM) model predicts that <strong>Heart Disease is absent</strong>.</p>
                <p style="font-size:1.15rem;"><strong>Confidence Score: {(1 - prob_heart_disease) * 100:.1f}% (No Heart Disease)</strong></p>
                <hr style="border-color: var(--green); opacity: 0.2;" />
                <p style="font-size:0.85rem; margin-bottom:0;"><em>Disclaimer: This assessment is based on a statistical machine learning model. It is for educational/screening purposes and does not substitute for a professional clinical diagnosis. Please consult a cardiologist or medical practitioner.</em></p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    try:
        df_clean = load_dataset()
        
        st.subheader("Explore the Training Dataset")
        st.write("Analyze patterns and feature relationships from the patients training dataset (180 cases).")
        
        # Metrics Row
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            metric_card("Total Patients", str(len(df_clean)))
        with mc2:
            hd_rate = df_clean["heart_disease_present"].mean() * 100
            metric_card("Heart Disease Rate", f"{hd_rate:.1f}%")
        with mc3:
            avg_age = df_clean["age"].mean()
            metric_card("Average Age", f"{avg_age:.1f} yrs")
        with mc4:
            male_pct = df_clean["sex"].mean() * 100
            metric_card("Male Patients", f"{male_pct:.1f}%")
            
        st.write("")
        st.write("---")
        st.write("")
        
        # Visualizer Panel
        viz_col1, viz_col2 = st.columns([3, 7])
        
        with viz_col1:
            st.write("#### Chart Controls")
            x_axis = st.selectbox(
                "X-Axis Column",
                ["age", "resting_blood_pressure", "serum_cholesterol_mg_per_dl", "max_heart_rate_achieved", "oldpeak_eq_st_depression", "chest_pain_type"]
            )
            y_axis = st.selectbox(
                "Y-Axis Column (Scatter/Box)",
                ["serum_cholesterol_mg_per_dl", "resting_blood_pressure", "max_heart_rate_achieved", "age", "oldpeak_eq_st_depression"]
            )
            plot_type = st.radio("Visualization Style", ["Scatter Plot", "Box Plot", "Histogram"])
            
        with viz_col2:
            st.markdown(f"""
            <div class="chart-wrap">
                <div class="chart-title">Patient Data Distribution</div>
                <div class="chart-subtitle">Analyzing {x_axis} vs {y_axis if plot_type != "Histogram" else "distribution"}</div>
            """, unsafe_allow_html=True)
            
            # Map target to nice labels for colors
            plot_df = df_clean.copy()
            plot_df["Heart Disease"] = plot_df["heart_disease_present"].map({0: "Absent", 1: "Present"})
            
            if plot_type == "Scatter Plot":
                fig = px.scatter(
                    plot_df,
                    x=x_axis,
                    y=y_axis,
                    color="Heart Disease",
                    color_discrete_map={"Absent": "#22c55e", "Present": "#ef4444"},
                    category_orders={"Heart Disease": ["Absent", "Present"]},
                    opacity=0.8
                )
            elif plot_type == "Box Plot":
                fig = px.box(
                    plot_df,
                    x="Heart Disease",
                    y=y_axis,
                    color="Heart Disease",
                    color_discrete_map={"Absent": "#22c55e", "Present": "#ef4444"},
                    category_orders={"Heart Disease": ["Absent", "Present"]}
                )
            else:
                fig = px.histogram(
                    plot_df,
                    x=x_axis,
                    color="Heart Disease",
                    barmode="group",
                    color_discrete_map={"Absent": "#22c55e", "Present": "#ef4444"},
                    category_orders={"Heart Disease": ["Absent", "Present"]}
                )
                
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Could not load data visualizations: {e}")

with tab3:
    st.subheader("Model Overview & Research details")
    
    st.write("""
    ### Machine Learning Pipeline
    
    1. **Data Preprocessing & Cleaning**:
       - Handled outlier cap/floors on continuous fields (`resting_blood_pressure`, `serum_cholesterol_mg_per_dl`).
       - Applied a log transformation (`np.log1p`) on the right-skewed `oldpeak_eq_st_depression` parameter to normalize its variance.
       - One-hot encoded categorical columns (`thal`).
       
    2. **Feature Scaling**:
       - Fit and scaled continuous inputs (`age`, `resting_blood_pressure`, `serum_cholesterol_mg_per_dl`, `max_heart_rate_achieved`, `oldpeak_eq_st_depression`) using `StandardScaler` to bring feature variances to equal scale.
       
    3. **Model Selection**:
       - Analyzed SVM, Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, and XGBoost.
       - The best performing model was a **Support Vector Classifier (SVC)** with an **RBF (Radial Basis Function) Kernel**, hyperparameter tuned using Randomized Search Cross-Validation (5-fold) optimizing for ROC AUC.
       
    4. **Model Performance**:
       - **Validation Accuracy**: `~83.3%`
       - **ROC AUC Score**: `~88.9%`
    """)
    
    st.write("---")
    
    st.write("""
    ### Feature Definitions
    
    - **Age**: Patient's age in years.
    - **Sex**: Patient's biological sex (Male or Female).
    - **Resting Blood Pressure**: Resting blood pressure in mm Hg on admission to the hospital.
    - **Serum Cholesterol**: Serum cholesterol level in mg/dl.
    - **Max Heart Rate**: Maximum heart rate achieved during stress test (bpm).
    - **Chest Pain Type**:
      - *Type 1*: Typical Angina
      - *Type 2*: Atypical Angina
      - *Type 3*: Non-anginal pain
      - *Type 4*: Asymptomatic chest pain
    - **Thal**: Thalassemia blood condition (Normal, Fixed Defect, or Reversible Defect).
    - **Exercise Induced Angina**: Angina (chest pain) induced by exercise.
    - **Oldpeak**: ST depression induced by exercise relative to rest.
    - **Slope of Peak Exercise ST Segment**:
      - *Type 1*: Upsloping
      - *Type 2*: Flat
      - *Type 3*: Downsloping
    - **Number of Major Vessels**: Number of major blood vessels (0-3) colored by fluoroscopy.
    - **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl (True/False).
    - **Resting EKG Results**: Resting electrocardiographic results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy).
    """)
