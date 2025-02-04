import streamlit as st
import pickle
import pandas as pd
import seaborn as sns

# Load trained model
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Streamlit UI
st.title("Tip Prediction App")

total_bill = st.number_input("Total Bill", min_value=0.0, step=0.1)
time = st.selectbox("Time", ["Lunch", "Dinner"])
time_encoded = 1 if time == "Dinner" else 0
size = st.number_input("Size", min_value=1, step=1)

if st.button("Predict Tip"):
    input_data = pd.DataFrame([[total_bill, time_encoded, size]], columns=['total_bill', 'time', 'size'])
    prediction = loaded_model.predict(input_data)
    st.write(f"Predicted Tip: ${prediction[0]:.2f}")
