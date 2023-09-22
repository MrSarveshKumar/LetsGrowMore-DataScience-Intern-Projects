import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained MNIST model (pickled)
model_filename = 'MNIST_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to preprocess the input image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image)  # Convert to NumPy array
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    image = image / 255.0  # Normalize pixel values
    return image

st.title('MNIST Digit Classifier')

uploaded_image = st.file_uploader("Upload a digit image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    if st.button("Predict"):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        digit = np.argmax(prediction)
        st.success(f"Predicted Digit: {digit}")


st.markdown("****")

st.write("NOTE: This is only for Educational Purpose")
st.markdown("<span style='font-size: 15px;'>Founder: *Sarvesh Kumar*</span>", unsafe_allow_html=True)
