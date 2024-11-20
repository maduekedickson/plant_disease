import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Streamlit app
st.title("Potato Disease Classifier")
st.write("Upload an image of a potato leaf to classify its health condition.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Predict the class
    predicted_class, confidence = predict(image)

    # Display results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence}%")
