import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import date
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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
    st.title("Singapore Resale Flat Prices Predicting")
    st.markdown(
        "The aim of this project is to create a machine learning model and implement it into an intuitive online application that forecasts Singaporean apartment prices for resale. The purpose of this predictive model is to help prospective buyers and sellers estimate the resale value of a flat. It is based on past data of resale flat transactions. Reason for Motivation: It might be difficult to determine the exact resale value of a flat in Singapore due to the fierce competition in the resale flat market. Resale values can be influenced by a wide range of variables, including location, apartment type, floor space, and length of lease. By giving customers a predicted resale price based on these variables, a predictive model can assist in overcoming these difficulties."
    )

if selected == 'Resale_prediction':
    # Define data types for reading the CSV file
    dtype_dict = {
        'month': 'float32',
        'year': 'float32',
        'town': 'category',
        'flat_type': 'category',
        'street_name': 'category',
        'floor_area_sqm': 'float32',
        'flat_model': 'category',
        'storey_start': 'float32',
        'storey_end': 'float32',
        'lease_commence_date': 'float32',
        'remaining_lease_months': 'float32',
        'resale_price': 'float32'
    }

    # Function to process each chunk
    def process_chunk(chunk):
        chunk.fillna({
            'month': chunk['month'].median(),
            'year': chunk['year'].median(),
            'storey_start': chunk['storey_start'].median(),
            'storey_end': chunk['storey_end'].median(),
            'lease_commence_date': chunk['lease_commence_date'].median(),
            'remaining_lease_months': chunk['remaining_lease_months'].median()
        }, inplace=True)
        
        # Convert columns to appropriate types
        chunk = chunk.astype({
            'month': 'int8',
            'year': 'int16',
            'storey_start': 'int8',
            'storey_end': 'int8',
            'lease_commence_date': 'int16',
            'remaining_lease_months': 'int16'
        })
        
        return chunk

    # Read and process data in chunks
    df_chunks = pd.read_csv('Singapore.csv.gz', dtype=dtype_dict, chunksize=1000)
    processed_chunks = [process_chunk(chunk) for chunk in df_chunks]

    # Concatenate processed chunks into one DataFrame
    df = pd.concat(processed_chunks)
    del df['Unnamed: 0']

    # Free up memory
    del df_chunks, processed_chunks

    # Input fields
    month = st.number_input("Month", min_value=1, max_value=12, step=1)
    current_year = date.today().year
    year = st.number_input("Year", min_value=1990, max_value=current_year, step=1)
    town = st.selectbox("Town", df['town'].unique())
    flat_type = st.selectbox("Flat Type", df['flat_type'].unique())
    street_name = st.selectbox("Street Name", df['street_name'].unique())
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=200.0, step=1.0)
    flat_model = st.selectbox("Flat Model", df['flat_model'].unique())
    storey_start = st.number_input("Storey Start", min_value=1, max_value=40, step=1)
    storey_end = st.number_input("Storey End", min_value=1, max_value=40, step=1)
    lease_commence_date = st.number_input("Lease Commencement Year", min_value=1960, max_value=current_year, step=1)
    remaining_lease_months = st.number_input("Remaining Lease (months)", min_value=1, max_value=1200, step=1)

    # Encode categorical variables using LabelEncoder
    label_encoders = {}
    categorical_cols = ['town', 'flat_type', 'street_name', 'flat_model']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    def load_model():
        # Split the data
        x = df.drop('resale_price', axis=1)
        y = df['resale_price']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

        # Model
        model = DecisionTreeRegressor(max_depth=25)
        model.fit(x_train, y_train)
        return model

    if st.button("Predict Selling Price"):
        # Load the model
        model = load_model()

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
        for col in categorical_cols:
            le = label_encoders[col]
            input_data[col] = le.transform(input_data[col])

        # Making predictions
        prediction = model.predict(input_data)

        # Display the prediction
        st.subheader(f"Predicted Selling Price: ${prediction[0]:,.2f}")
