import streamlit as st
import tensorflow as tf
import keras.utils as image
import numpy as np
from PIL import Image

# Load the trained model (assuming it's saved as 'model.h5')
model = tf.keras.models.load_model('cnn_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((64, 64))  # Resize to the size the model expects
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Scale the image
    return img_array

# Function to make predictions
def predict_image(img_array):
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Cat 😺"
    else:
        return "Dog 🐶"

#App title
favicon = Image.open('favicon.png')
st.set_page_config(page_title="Dog vs Cat", page_icon= favicon,layout="wide", menu_items=None)

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
</style>
""",unsafe_allow_html=True)

css = '''
<style>
section.main > div:has(~ footer ) {
    padding-bottom: 0;
}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# Streamlit interface
left, right = st.columns(2,gap="medium") 

with left :
    st.title("Dog vs Cat Image Classification", anchor = False)
    st.write("Upload an image to classify it as a dog or a cat.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with right: 
    if uploaded_file is not None:
        img_array = preprocess_image(uploaded_file)
        prediction = predict_image(img_array)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width="auto")
        with left:
            st.header(f"Prediction: {prediction}", anchor = False)