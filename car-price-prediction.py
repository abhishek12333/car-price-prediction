import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("Car details v3.csv")

# Data Preprocessing
data.drop(columns=["name", "torque"], inplace=True, errors="ignore")
data["mileage"] = pd.to_numeric(data["mileage"].str.extract(r'([\d\.]+)')[0], errors="coerce")
data["engine"] = pd.to_numeric(data["engine"].str.extract(r'([\d\.]+)')[0], errors="coerce")
data["max_power"] = pd.to_numeric(data["max_power"].str.extract(r'([\d\.]+)')[0], errors="coerce")
data.fillna(data.median(numeric_only=True), inplace=True)
if "seats" in data.columns:
    data["seats"] = data["seats"].astype(int, errors="ignore")
data = pd.get_dummies(data, columns=["fuel", "seller_type", "transmission", "owner"], drop_first=True)

# Split data
X = data.drop(columns=["selling_price"])
y = data["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Streamlit UI
st.title("Car Price Prediction App")
st.write("Enter the car details to predict its selling price.")

# User input fields
user_input = {}
for column in X.columns:
    user_input[column] = st.number_input(f"Enter {column}", value=float(X[column].median()))

data_point = pd.DataFrame([user_input])

if st.button("Predict Price"):
    prediction = model.predict(data_point)
    st.success(f"Estimated Selling Price: ₹{prediction[0]:,.2f}")

# Display model performance
st.subheader("Model Performance")
y_pred = model.predict(X_test)
st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
st.write(f"Mean Absolute Error: ₹{mean_absolute_error(y_test, y_pred):,.2f}")
