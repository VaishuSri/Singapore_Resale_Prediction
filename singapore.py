import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import date
import joblib

# Streamlit part
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    selected = option_menu("Menu", ["Home", "Resale_prediction"],
                           icons=["house", "graph-up"],
                           menu_icon="menu-button-wide",
                           )

if selected == 'Home':
    st.title("Singapore Resale Flat Prices Prediction")
    st.markdown(
        """
        The aim of this project is to create a machine learning model and implement it into an intuitive online application that forecasts Singaporean apartment prices for resale. 
        The purpose of this predictive model is to help prospective buyers and sellers estimate the resale value of a flat. It is based on past data of resale flat transactions. 
        Resale values can be influenced by a wide range of variables, including location, apartment type, floor space, and length of lease. By giving customers a predicted resale price based on these variables, 
        a predictive model can assist in overcoming these difficulties.
        """
    )

if selected == 'Resale_prediction':
    # Load serialized objects
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoders = {col: joblib.load(f'{col}_encoder.joblib') for col in ['town', 'flat_type', 'street_name', 'flat_model']}

    # Read a sample of the dataset for select box options
    sample_df = pd.read_csv('Singapore.csv.gz', nrows=1000)

    # Input fields
    month = st.number_input("Month", min_value=1, max_value=12, step=1)
    current_year = date.today().year
    year = st.number_input("Year", min_value=1990, max_value=current_year, step=1)
    town = st.selectbox("Town", sample_df['town'].unique())
    flat_type = st.selectbox("Flat Type", sample_df['flat_type'].unique())
    street_name = st.selectbox("Street Name", sample_df['street_name'].unique())
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=200.0, step=1.0)
    flat_model = st.selectbox("Flat Model", sample_df['flat_model'].unique())
    storey_start = st.number_input("Storey Start", min_value=1, max_value=40, step=1)
    storey_end = st.number_input("Storey End", min_value=1, max_value=40, step=1)
    lease_commence_date = st.number_input("Lease Commencement Year", min_value=1960, max_value=current_year, step=1)
    remaining_lease_months = st.number_input("Remaining Lease (months)", min_value=1, max_value=1200, step=1)

    if st.button("Predict Selling Price"):
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'month': [month],
            'year': [year],
            'town': [town],
            'flat_type': [flat_type],
            'street_name': [street_name],
            'floor_area_sqm': [floor_area_sqm],
            'flat_model': [flat_model],
            'storey_start': [storey_start],
            'storey_end': [storey_end],
            'lease_commence_date': [lease_commence_date],
            'remaining_lease_months': [remaining_lease_months]
        })

        # Encode input data using previously fitted label encoders
        for col in ['town', 'flat_type', 'street_name', 'flat_model']:
            le = label_encoders[col]
            input_data[col] = le.transform(input_data[col])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make predictions
        prediction = model.predict(input_data_scaled)

        # Display the prediction
        st.subheader(f"Predicted Selling Price: ${prediction[0]:,.2f}")

