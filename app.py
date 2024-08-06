import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app for user input and prediction
def main():
    st.title("Iris Flower Species Prediction")

    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

    if st.button('Predict'):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        st.write(f"The predicted species is: {prediction[0]}")

if __name__ == '__main__':
    main()
