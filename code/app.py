import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import os


st.set_page_config(page_title="Fashion MNIST Classifier", page_icon="👕")

st.title(" Fashion-MNIST AI Classifier")
st.write("Upload an image of a clothing item, and the AI will identify it!")

@st.cache_resource
def load_my_model():

    model_path = os.path.join("saved_model", "best_model.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None


model = load_my_model()

if model is None:
    st.error(" Error: Model file not found. Please train the model first!")
else:
    st.success(" Model loaded successfully!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


def process_image(image_data):

    img = ImageOps.grayscale(image_data)

    img = img.resize((28, 28))

    img_array = np.array(img)

    img_array = img_array.astype('float32') / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array



if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, width=200)

    if st.button(' Identify Item'):

        processed_img = process_image(image)

        prediction = model.predict(processed_img)
        class_index = np.argmax(prediction)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        result = class_names[class_index]
        confidence = np.max(prediction) * 100

        st.markdown(f"###  It's a: **{result}**")
        st.info(f"Confidence: {confidence:.2f}%")

        st.bar_chart(prediction[0])