import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Page Settings
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

# Title
st.title("House Price Prediction")
st.write("Enter house details to predict the estimated price.")

# Create Dataset
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Area": np.random.randint(500, 5000, n),
    "Bedrooms": np.random.randint(1, 6, n),
    "Bathrooms": np.random.randint(1, 5, n),
    "Parking": np.random.randint(0, 4, n),
    "Floors": np.random.randint(1, 4, n),
    "Age": np.random.randint(0, 30, n)
})

# Calculate House Price
data["Price"] = (
    data["Area"] * 4500
    + data["Bedrooms"] * 400000
    + data["Bathrooms"] * 250000
    + data["Parking"] * 150000
    + data["Floors"] * 200000
    - data["Age"] * 50000
)

# Features and Target
X = data[
    ["Area", "Bedrooms", "Bathrooms",
     "Parking", "Floors", "Age"]
]

y = data["Price"]

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# User Input
st.header("Enter House Details")

area = st.number_input(
    "Area (Square Feet)",
    min_value=500,
    max_value=10000,
    value=1500
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=5,
    value=1
)

floors = st.number_input(
    "Number of Floors",
    min_value=1,
    max_value=5,
    value=1
)

age = st.number_input(
    "House Age (Years)",
    min_value=0,
    max_value=100,
    value=5
)

# Prediction
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Parking": [parking],
        "Floors": [floors],
        "Age": [age]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated House Price: ₹{prediction:,.0f}"
    )

    st.subheader("House Details")

    st.write(f"Area: {area} sq ft")
    st.write(f"Bedrooms: {bedrooms}")
    st.write(f"Bathrooms: {bathrooms}")
    st.write(f"Parking: {parking}")
    st.write(f"Floors: {floors}")
    st.write(f"House Age: {age} years")