# Car Price Prediction using Linear Regression

## 📌 Overview
This project builds a **Car Price Prediction Model** using **Linear Regression**. The dataset is preprocessed, trained, and evaluated to predict car selling prices based on various features. The project is visualized using **Streamlit** and deployed on **GitHub**.

## 📂 Dataset
The dataset used is **Car details v3.csv** (sourced from Kaggle). It contains various attributes related to cars, such as:
- **Mileage**
- **Engine Power**
- **Max Power**
- **Fuel Type**
- **Transmission**
- **Owner Type**
- **Seats**
- **Selling Price** (Target Variable)

## 🛠️ Steps in the Project
### **1. Data Preprocessing**
- **Dropped unnecessary columns**: `name`, `torque`
- **Converted categorical data**: One-hot encoding for `fuel`, `seller_type`, `transmission`, `owner`
- **Converted numerical data**: Extracted numeric values from `mileage`, `engine`, `max_power`
- **Handled missing values**: Filled missing values using the median
- **Converted seats to integer**

### **2. Model Training**
- Split data into **80% training** and **20% testing**
- Used **Linear Regression** to train the model
- Saved the trained model using `pickle`

### **3. Model Evaluation**
- Used **R² Score** and **Mean Absolute Error (MAE)** for performance evaluation

## 🚀 How to Run the Project
### **1. Install Dependencies**
```bash
pip install pandas scikit-learn pickle-mixin streamlit
