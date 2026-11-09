import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "flight_price_prediction_pipeline.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("✈️ Flight Price Prediction")
st.markdown(
    "### Predict your estimated flight ticket price"
)

st.write(
    "Enter your journey details below to get an estimated flight price."
)

st.divider()

# -------------------------------------------------
# FLIGHT INPUT OPTIONS
# -------------------------------------------------

airlines = [
    "Air Asia",
    "Air India",
    "GoAir",
    "IndiGo",
    "Jet Airways",
    "Jet Airways Business",
    "Multiple carriers",
    "Multiple carriers Premium economy",
    "SpiceJet",
    "Trujet",
    "Vistara",
    "Vistara Premium economy"
]

cities = [
    "Banglore",
    "Chennai",
    "Delhi",
    "Kolkata",
    "Mumbai"
]

destinations = [
    "Banglore",
    "Cochin",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "New Delhi"
]

# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------

st.subheader("Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Select Airline",
        airlines
    )

    source = st.selectbox(
        "Source City",
        cities
    )

    journey_date = st.date_input(
        "Journey Date"
    )

    departure_time = st.time_input(
        "Departure Time"
    )

with col2:
    destination = st.selectbox(
        "Destination City",
        destinations
    )

    total_stops = st.selectbox(
        "Total Stops",
        [0, 1, 2, 3, 4]
    )

    arrival_time = st.time_input(
        "Arrival Time"
    )

    duration_hours = st.number_input(
        "Flight Duration (Hours)",
        min_value=0.5,
        max_value=40.0,
        value=2.5,
        step=0.5
    )

st.divider()

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button(
    "🔍 Predict Flight Price",
    use_container_width=True
):

    if source == destination:
        st.error(
            "Source and destination cannot be the same."
        )

    else:

        input_data = pd.DataFrame([{
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Total_Stops": total_stops,
            "Journey_Day": journey_date.day,
            "Journey_Month": journey_date.month,
            "Dep_Hour": departure_time.hour,
            "Dep_Minute": departure_time.minute,
            "Arrival_Hour": arrival_time.hour,
            "Arrival_Minute": arrival_time.minute,
            "Duration_Hours": duration_hours,
            "Journey_Weekday": journey_date.weekday()
        }])

        try:
            prediction = model.predict(input_data)[0]

            st.success("Prediction Completed!")

            st.subheader("Estimated Flight Price")

            st.metric(
                label="Predicted Ticket Price",
                value=f"₹{prediction:,.2f}"
            )

            with st.expander("View Flight Details"):
                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Developed using Python, Streamlit and XGBoost."
)

st.caption(
    "Note: This application provides dataset-based "
    "estimated prices, not real-time airline fares."
)
