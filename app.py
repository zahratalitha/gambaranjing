import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.efficientnet import preprocess_input

repo_id = "zahratalitha/klasifikasikucing"   
filename = "kucinganjing_full.h5"  
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
model = tf.keras.models.load_model(model_path)

class_names = ["Kucing 🐱", "Anjing 🐶"]

st.set_page_config(page_title="Klasifikasi Anjing vs Kucing 🐾", page_icon="🐶")
st.title("🐾 Klasifikasi Anjing vs Kucing")
st.write("Upload gambar")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)
    img_height, img_width = 224, 224 
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.image.resize(img_array, [img_height, img_width])
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.subheader("📊 Hasil Prediksi")
    st.write(
        f"👉 Gambar ini kemungkinan besar adalah **{class_names[np.argmax(score)]}** "
        f"dengan tingkat keyakinan **{100 * np.max(score):.2f}%**"
    )
