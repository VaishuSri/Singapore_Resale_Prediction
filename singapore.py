import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from datetime import date
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Streamlit part
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache(allow_output_mutation=True)
def load_data():
    # Read data in chunks to reduce memory usage
    df_chunks = pd.read_csv('Singapore.csv.gz', chunksize=1000)

    # Concatenate chunks into one DataFrame
    df = pd.concat(df_chunks)
    del df['Unnamed: 0']

    return df

df = load_data()

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
    df[col] = df[col].astype('category')
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Split the data
x = df.drop(columns=['resale_price'])
y = df['resale_price']

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['floor_area_sqm', 'storey_start', 'storey_end', 'lease_commence_date', 'remaining_lease_months']
x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

# Model
model = DecisionTreeRegressor(max_depth=25)
model.fit(x_train, y_train)

if st.button("Predict Selling Price"):
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

    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform([input_data[col].iloc[0]])

    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    prediction = model.predict(input_data)

    st.subheader(f"Predicted Selling Price: ${prediction[0]:,.2f}")


