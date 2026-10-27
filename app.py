"""TrafficSense AI — Traffic Congestion Prediction Using Artificial Neural Networks (ANN).

Streamlit Web Application for interactive demonstration and model evaluation.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# Application paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "traffic_sense_ann.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
OUTPUT_DIR = ROOT / "outputs"
CLASSES = ["Low Congestion", "Moderate Congestion", "High Congestion"]

# Page layout & styling
st.set_page_config(
    page_title="TrafficSense AI — Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header banner
st.title("🚦 TrafficSense AI")
st.subheader("Traffic Congestion Prediction Using Artificial Neural Networks (ANN)")
st.caption(
    "A machine learning application predicting interstate traffic congestion levels "
    "from temporal, meteorological, and calendar features using a Scikit-learn MLPClassifier."
)

# Sidebar with summary information
with st.sidebar:
    st.header("📌 Project Details")
    st.markdown("**Author:** MBA in Data Science & AI")
    st.markdown("**Architecture:** Multi-Layer Perceptron (ANN)")
    st.markdown("**Dataset:** UCI Metro Interstate Traffic (I-94)")
    st.markdown("**Splitting Strategy:** Chronological 70/15/15")
    
    st.divider()
    st.header("🏷️ Congestion Classes")
    st.markdown(
        "- 🟢 **Low Congestion**: Volume ≤ 2,154 vehicles/hr\n"
        "- 🟡 **Moderate Congestion**: 2,154 < Volume ≤ 4,555 vehicles/hr\n"
        "- 🔴 **High Congestion**: Volume > 4,555 vehicles/hr"
    )
    st.caption("Thresholds derived from training-period traffic volume terciles (quantiles).")

# Navigation tabs
overview_tab, predict_tab, analysis_tab, model_tab = st.tabs([
    "🏠 Project Overview",
    "🔮 Interactive Prediction",
    "📊 Dataset & Visualizations",
    "🧠 Model Architecture & Metrics",
])

# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------
with overview_tab:
    st.markdown("### 🎯 Executive Summary & Objectives")
    st.write(
        "Urban traffic congestion causes billions of dollars in lost productivity, excess fuel consumption, "
        "and increased carbon emissions annually. **TrafficSense AI** is a machine learning solution that "
        "predicts highway traffic congestion levels based on time, calendar events, and weather conditions. "
        "By applying an **Artificial Neural Network (ANN)**, the system models non-linear relationships between "
        "environmental drivers and traffic flow."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🟢 **Low Congestion**\n\nVolume ≤ 2,154 veh/hr.\n\nFree-flow traffic conditions with minimal delays.")
    with col2:
        st.warning("🟡 **Moderate Congestion**\n\n2,154 < Volume ≤ 4,555 veh/hr.\n\nSteady traffic flow with minor speed reductions.")
    with col3:
        st.error("🔴 **High Congestion**\n\nVolume > 4,555 veh/hr.\n\nHeavy traffic volume, peak rush hour congestion.")

    st.markdown("### 🛠️ Technology Stack")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Language", "Python 3")
    col_t2.metric("ML Model", "Scikit-Learn MLP")
    col_t3.metric("Data Engine", "Pandas & NumPy")
    col_t4.metric("UI Framework", "Streamlit")

    st.markdown("### 🔄 End-to-End Workflow")
    st.markdown(
        """
        1. **Data Ingestion**: 48,187 hourly observations (2012–2018) from the UCI I-94 dataset.
        2. **Feature Engineering**: Time extraction (hour, day of week, month), holiday detection, and weather categorization.
        3. **Chronological Splitting**: 70% Training, 15% Validation, and 15% Hold-out Testing (prevents future-to-past data leakage).
        4. **Scikit-Learn Pipeline**: Median imputation & `StandardScaler` for numeric features; `OneHotEncoder` for categories.
        5. **ANN Model**: Multi-Layer Perceptron (64, 32 neurons) with ReLU activation, Adam optimizer, and early stopping.
        6. **Interactive Deployment**: Real-time inference via this Streamlit dashboard.
        """
    )

# ---------------------------------------------------------
# TAB 2: INTERACTIVE PREDICTION
# ---------------------------------------------------------
with predict_tab:
    st.markdown("### 🔮 Predict Traffic Congestion Level")
    st.write(
        "Select scenario conditions below. The inputs will be transformed through the trained preprocessing pipeline "
        "and passed to the Artificial Neural Network to generate a prediction and probability distribution."
    )

    # Preset Quick-Select buttons for viva demonstration
    st.markdown("##### ⚡ Quick Demonstration Presets:")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    
    preset_hour = 8
    preset_day = 1
    preset_month = 6
    preset_weather = "Clear"
    preset_temp_c = 20.0
    preset_rain = 0.0
    preset_snow = 0.0
    preset_clouds = 20
    preset_holiday = "None"

    if "preset" not in st.session_state:
        st.session_state.preset = None

    if q_col1.button("🚗 Tuesday Morning Rush (8 AM)"):
        st.session_state.preset = "rush_morning"
    if q_col2.button("🌙 Midnight Travel (2 AM)"):
        st.session_state.preset = "night"
    if q_col3.button("☀️ Saturday Afternoon (3 PM)"):
        st.session_state.preset = "weekend"
    if q_col4.button("❄️ Winter Snowstorm (5 PM)"):
        st.session_state.preset = "snowstorm"

    if st.session_state.preset == "rush_morning":
        preset_hour, preset_day, preset_month = 8, 1, 10
        preset_weather, preset_temp_c, preset_clouds = "Clear", 12.0, 10
    elif st.session_state.preset == "night":
        preset_hour, preset_day, preset_month = 2, 2, 7
        preset_weather, preset_temp_c, preset_clouds = "Clear", 18.0, 0
    elif st.session_state.preset == "weekend":
        preset_hour, preset_day, preset_month = 15, 5, 6
        preset_weather, preset_temp_c, preset_clouds = "Clouds", 22.0, 50
    elif st.session_state.preset == "snowstorm":
        preset_hour, preset_day, preset_month = 17, 3, 1
        preset_weather, preset_temp_c, preset_snow, preset_clouds = "Snow", -5.0, 2.5, 95

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("##### ⏰ Time & Calendar")
            hour = st.slider("Hour of Day (0–23)", 0, 23, int(preset_hour), help="Hour in 24-hour format (e.g., 8 = 8 AM, 17 = 5 PM)")
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day = st.selectbox("Day of Week", list(range(7)), index=int(preset_day), format_func=lambda x: day_names[x])
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            month = st.selectbox("Month of Year", list(range(1, 13)), index=int(preset_month - 1), format_func=lambda x: f"{x} - {month_names[x-1]}")
            holiday = st.text_input("Holiday Name (or 'None')", value=preset_holiday, help="Enter 'None' for regular working days, or names like 'Thanksgiving Day', 'Labor Day', etc.")

        with c2:
            st.markdown("##### 🌦️ Weather Conditions")
            weather_options = ["Clear", "Clouds", "Rain", "Drizzle", "Mist", "Fog", "Snow", "Haze", "Thunderstorm", "Squall", "Smoke"]
            weather_idx = weather_options.index(preset_weather) if preset_weather in weather_options else 0
            weather = st.selectbox("Weather Category", weather_options, index=weather_idx)
            temp_c = st.number_input("Temperature (°C)", min_value=-40.0, max_value=50.0, value=float(preset_temp_c), step=1.0)
            temp_kelvin = temp_c + 273.15
            st.caption(f"Equivalent Kelvin: **{temp_kelvin:.2f} K** (used by model)")

        with c3:
            st.markdown("##### 💧 Precipitation & Sky")
            rain = st.number_input("Rain in Past Hour (mm)", min_value=0.0, max_value=300.0, value=float(preset_rain), step=0.5)
            snow = st.number_input("Snow in Past Hour (mm)", min_value=0.0, max_value=300.0, value=float(preset_snow), step=0.5)
            clouds = st.slider("Cloud Cover (%)", 0, 100, int(preset_clouds))

        submit = st.form_submit_button("🚀 Run ANN Prediction", type="primary", use_container_width=True)

    if submit:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            st.error("Model files not found. Please train the model first by running `python train_model.py`.")
        else:
            try:
                pipeline = joblib.load(MODEL_PATH)
                input_df = pd.DataFrame([{
                    "hour": hour,
                    "day_of_week": day,
                    "month": month,
                    "temp": temp_kelvin,
                    "rain_1h": rain,
                    "snow_1h": snow,
                    "clouds_all": clouds,
                    "holiday": holiday.strip() if holiday.strip() else "None",
                    "weather_main": weather,
                }])
                
                prediction = str(pipeline.predict(input_df)[0])
                
                # Visual output card
                st.markdown("---")
                st.markdown("### 🏁 Prediction Output")
                
                p_col1, p_col2 = st.columns([1, 1])
                with p_col1:
                    if prediction == "Low Congestion":
                        st.success(f"### 🟢 Predicted: **{prediction}**")
                        st.write("✅ **Traffic Status:** Free-flowing traffic expected. Ideal travel conditions.")
                    elif prediction == "Moderate Congestion":
                        st.warning(f"### 🟡 Predicted: **{prediction}**")
                        st.write("⚠️ **Traffic Status:** Steady volume with possible localized slowing.")
                    else:
                        st.error(f"### 🔴 Predicted: **{prediction}**")
                        st.write("🛑 **Traffic Status:** High volume / peak congestion anticipated. Delays likely.")

                with p_col2:
                    if hasattr(pipeline, "predict_proba"):
                        probabilities = pipeline.predict_proba(input_df)[0]
                        prob_df = pd.DataFrame({
                            "Congestion Level": pipeline.classes_,
                            "Confidence": probabilities
                        }).set_index("Congestion Level")
                        st.markdown("##### Model Confidence Distribution")
                        st.bar_chart(prob_df)
                        
                        top_conf = max(probabilities) * 100
                        st.caption(f"Model assigns **{top_conf:.1f}% confidence** to the winning class.")

            except Exception as e:
                st.error(f"Prediction encountered an error: {e}")

# ---------------------------------------------------------
# TAB 3: DATA ANALYSIS & CHARTS
# ---------------------------------------------------------
with analysis_tab:
    st.markdown("### 📊 Dataset Exploration & Analysis Charts")
    st.write(
        "Visualizations generated from actual historical data of the I-94 interstate corridor (Minneapolis–St. Paul). "
        "These charts reveal the key temporal and volume patterns discovered during exploratory data analysis."
    )

    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        cat_chart = OUTPUT_DIR / "category_distribution.png"
        if cat_chart.exists():
            st.markdown("#### 1. Target Class Distribution")
            st.image(str(cat_chart), use_container_width=True)
            st.caption("Distribution across Low, Moderate, and High classes derived from training quantiles.")
        else:
            st.info("Chart not found. Run `python train_model.py` to generate charts.")

    with chart_col2:
        hour_chart = OUTPUT_DIR / "traffic_by_hour.png"
        if hour_chart.exists():
            st.markdown("#### 2. Hourly Traffic Volume Trend")
            st.image(str(hour_chart), use_container_width=True)
            st.caption("Diurnal pattern showing peak travel demand at morning (7-9 AM) and evening (4-6 PM) rush hours.")
        else:
            st.info("Chart not found. Run `python train_model.py` to generate charts.")

    st.markdown("#### 3. Confusion Matrix (Evaluation on Unseen Test Data)")
    cm_chart = OUTPUT_DIR / "confusion_matrix.png"
    if cm_chart.exists():
        st.image(str(cm_chart), use_container_width=True)
        st.caption("Diagonal elements represent correct predictions on the chronologically held-out test set.")

# ---------------------------------------------------------
# TAB 4: MODEL INFORMATION & METRICS
# ---------------------------------------------------------
with model_tab:
    st.markdown("### 🧠 Neural Network Architecture & Evaluation")
    
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Hold-out Test Accuracy", f"{meta['accuracy']:.2%}")
        m_col2.metric("Validation Accuracy", f"{meta['validation_accuracy']:.2%}")
        m_col3.metric("Total Rows", f"{meta['rows_after_cleaning']:,}")
        m_col4.metric("ANN Iterations (Epochs)", meta.get("iterations", "N/A"))

        st.markdown("#### 📋 Detailed Classification Performance")
        report = meta["classification_report"]
        report_data = []
        for cls in ["Low Congestion", "Moderate Congestion", "High Congestion"]:
            if cls in report:
                report_data.append({
                    "Class": cls,
                    "Precision": f"{report[cls]['precision']:.2%}",
                    "Recall": f"{report[cls]['recall']:.2%}",
                    "F1-Score": f"{report[cls]['f1-score']:.2%}",
                    "Support (Test Count)": int(report[cls]["support"])
                })
        st.table(pd.DataFrame(report_data).set_index("Class"))

        st.markdown("#### ⚙️ ANN Hyperparameters & Setup")
        st.markdown(
            f"""
            - **Algorithm:** Multi-Layer Perceptron (`MLPClassifier`)
            - **Hidden Layers:** 2 layers (Layer 1: 64 neurons, Layer 2: 32 neurons)
            - **Activation Function:** Rectified Linear Unit (`ReLU`)
            - **Optimization Solver:** `Adam` (Adaptive Moment Estimation)
            - **L2 Regularization (`alpha`):** 0.001
            - **Initial Learning Rate:** 0.001
            - **Early Stopping:** Enabled (patience = 15 iterations without validation improvement)
            - **Training Volume Terciles:** Low: ≤ {meta['low_volume_threshold_inclusive']:.0f} | Moderate: ≤ {meta['moderate_volume_upper_threshold_inclusive']:.0f}
            """
        )

        st.markdown("#### ⚠️ Operational Limitations & Academic Context")
        st.markdown(
            """
            1. **Proxy Measurement**: The dataset measures vehicle volume count at a fixed interstate station, which acts as an operational proxy rather than a direct sensor of localized road speed or driver delay.
            2. **Geographic Specificity**: Trained on westbound I-94 in Minnesota. Travel patterns and weather effects may differ in other metropolitan areas.
            3. **Sensor Artifacts**: The raw dataset contains occasional extreme readings (such as 0 K temperature or sensor outages) handled during preprocessing.
            4. **Academic Purpose**: Designed as an educational demonstration of neural network classification, not a commercial live navigation service.
            """
        )
    else:
        st.warning("Model metadata file not found. Please run `python train_model.py` first.")
