import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="🐶🐱 Klasifikasi Gambar")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/klasifikasikucing",  
        filename="kucinganjing_full.h5"         
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

st.title("🐶🐱 Klasifikasi Gambar: Anjing vs Kucing")
uploaded_file = st.file_uploader("Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") 
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img = image.resize((180, 180))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1,180,180,3)

    prediction = model.predict(img_array)[0][0]
    prob = float(prediction)

    if prob > 0.5:
        st.success(f"Prediksi: 🐶 Anjing ({prob:.2%})")
    else:
        st.success(f"Prediksi: 🐱 Kucing ({(1-prob):.2%})")
