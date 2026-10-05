import streamlit as st
import joblib
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="DRI Predictor", 
    page_icon="⛏️", 
    layout="centered"
)

st.title("⛏️ Rock Drillability Index (DRI) Predictor")
st.write("Enter laboratory mechanical test properties to estimate the Drilling Rate Index (DRI):")

# 2. Load Model Pipeline and Feature Names
@st.cache_resource
def load_assets():
    model = joblib.load('best_rock_drillability_svr.pkl')
    data = pd.read_csv('rock_drillability_data.csv')
    feature_names = list(data.columns[:-1])
    return model, feature_names

model, feature_names = load_assets()

# 3. Dynamic Dual-Column Inputs
input_data = {}
col1, col2 = st.columns(2)

# Default values tailored for typical rock mechanical test ranges
default_vals = {
    'UCS_MPa': 120.0,
    'BTS_MPa': 10.0,
    'Schmidt_Rebound': 35.0,
    'Point_Load_MPa': 5.0,
    'P_Wave_m_s': 3500.0,
    'Density_g_cm3': 2.65
}

for i, feature in enumerate(feature_names):
    default_val = default_vals.get(feature, 50.0)
    with col1 if i % 2 == 0 else col2:
        input_data[feature] = st.number_input(
            label=f"{feature}",
            value=float(default_val),
            step=1.0
        )

# 4. Prediction Engine
st.markdown("---")
if st.button("🚀 Calculate DRI Prediction", use_container_width=True):
    # Pass inputs as DataFrame with matching column names
    input_df = pd.DataFrame([input_data])
    
    # Generate prediction using SVR pipeline
    prediction = model.predict(input_df)[0]
    
    # Display Result
    st.success(f"### Predicted Drilling Rate Index (DRI): **{prediction:.2f}**")

# 5. Model Insights & Visualization Section
st.markdown("---")
st.subheader("📊 Model Performance & Feature Insights")

if st.checkbox("Show Model Evaluation Visualizations"):
    try:
        st.image(
            'drillability_final_results.png', 
            caption='Left: XGBoost Relative Feature Importance | Right: Tuned SVR Actual vs. Predicted Correlation'
        )
    except FileNotFoundError:
        st.warning("Visualizations file ('drillability_final_results.png') not found. Run 'rock_drillability.py' first.")