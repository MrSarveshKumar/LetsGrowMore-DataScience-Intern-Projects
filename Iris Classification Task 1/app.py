import streamlit as st
import pandas as pd
import numpy as np
import pickle

model_file = 'iris_model.pkl'
with open(model_file, 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Iris Flower Species Prediction")

st.header("Input Features")
sepal_length = st.number_input('Enter the Sepal Length (cm)')
sepal_width = st.number_input('Enter the Sepal Width (cm)')
petal_length = st.number_input('Enter the Petal Length (cm)')
petal_width = st.number_input('Enter the Petal Width (cm)')

user_input = pd.DataFrame({
    'SepalLengthCm': [sepal_length],
    'SepalWidthCm': [sepal_width],
    'PetalLengthCm': [petal_length],
    'PetalWidthCm': [petal_width]
})

predicted_species = model.predict(user_input)

if st.button('Predict'):
    if predicted_species == 'Iris-setosa':
        st.success("The predicted species is: Iris-Setosa")
        st.image('static/iris-setosa.png')
    elif predicted_species == 'Iris-versicolor':
        st.success("The predicted species is: Iris-Versicolor")
        st.image('static/iris-versicolor.jpeg')
    elif predicted_species == 'Iris-virginica':
        st.success("The predicted species is: Iris-Virginica")
        st.image('static/iris-virginica.png')

st.markdown("****")

st.write("NOTE: This is only for Educational Purpose")
st.markdown("<span style='font-size: 15px;'>Founder: *Sarvesh Kumar*</span>", unsafe_allow_html=True)