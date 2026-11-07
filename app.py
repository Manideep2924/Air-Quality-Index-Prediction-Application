
# Import required libraries
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌍",
    layout="wide"
)


# 2. LOAD THE TRAINED MACHINE LEARNING MODEL


BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "aqi_model.joblib")
    feature_names = joblib.load(
        BASE_DIR / "aqi_feature_names.joblib"
    )
except FileNotFoundError:
    st.error("Model files not found. Please run train.py first.")
    st.stop()


# 3. DISPLAY THE MAIN HEADING


st.title("🌍 Air Quality Index Prediction Application")
st.write("Predict AQI using environmental measurements.")


# 4. CREATE THE SIDEBAR


st.sidebar.header("Environmental Information")

pm25 = st.sidebar.number_input(
    "PM2.5",
    min_value=0.0,
    value=25.0
)

pm10 = st.sidebar.number_input(
    "PM10",
    min_value=0.0,
    value=50.0
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=25.0
)

selected_date = st.sidebar.date_input(
    "Measurement Date"
)


# 5. SELECT LOCATION


location_columns = [
    name for name in feature_names
    if name.startswith("location_")
]

locations = [
    name.replace("location_", "")
    for name in location_columns
]

if locations:
    location = st.sidebar.selectbox(
        "Select Location",
        locations
    )
else:
    location = None
    st.warning("No location features found in the model.")


# 6. CREATE INPUT DATA


input_values = {
    "pm2.5": pm25,
    "pm10": pm10,
    "temperature": temperature,
    "year": selected_date.year,
    "month": selected_date.month,
    "day": selected_date.day
}

for name in feature_names:
    if name.startswith("location_"):
        input_values[name] = int(
            name == f"location_{location}"
        )

# Arrange the input features in the training order
input_df = pd.DataFrame(
    [[input_values[name] for name in feature_names]],
    columns=feature_names
)


# 7. DISPLAY ENVIRONMENTAL INFORMATION


st.subheader("Environmental Details")

st.dataframe(
    input_df,
    use_container_width=True
)


# 8. PREDICT AIR QUALITY INDEX


if st.button("Predict AQI", type="primary"):

    prediction = float(model.predict(input_df)[0])

    st.subheader("AQI Prediction Result")

    st.metric(
        label="Estimated AQI",
        value=f"{prediction:.2f}"
    )

    if prediction < 0:
        st.warning(
            "The model returned a negative AQI value. "
            "Please check the model and input data."
        )
    elif prediction <= 50:
        st.success("Category: Good")
    elif prediction <= 100:
        st.info("Category: Satisfactory")
    elif prediction <= 200:
        st.warning("Category: Moderate")
    elif prediction <= 300:
        st.warning("Category: Poor")
    elif prediction <= 400:
        st.error("Category: Very Poor")
    else:
        st.error("Category: Severe")

    st.caption(
        "This is a machine learning estimate, "
        "not an official AQI measurement."
    )