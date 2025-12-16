import streamlit as st
import pandas as pd
import joblib

# App title
st.title("Titanic Survival Prediction App")
st.write("Fill the information below to predict survival probability.")

# Load model and scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

# User inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 1, 90, 30)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Create dataframe
data = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [Sex],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Fare": [Fare],
    "Embarked": [Embarked]
})

# One-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Required columns (same as training)
required_cols = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
    'Sex_male', 'Embarked_Q', 'Embarked_S'
]

# Add missing columns
for col in required_cols:
    if col not in data.columns:
        data[col] = 0

data = data[required_cols]

# Scale input
data_scaled = scaler.transform(data)

# Prediction
prediction = model.predict(data_scaled)[0]
probability = model.predict_proba(data_scaled)[0][1]

# Output
st.subheader("Prediction Result")
st.write("Survived ✅" if prediction == 1 else "Did Not Survive ❌")

st.subheader("Survival Probability")
st.write(f"{probability:.3f}")