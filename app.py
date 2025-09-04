import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# --- Load model ---
@st.cache_resource
def load_model():
    repo_id = "zahratalitha/klasifikasikucing"   # ganti sesuai repo kamu
    filename = "kucinganjing_full.h5"            # nama file model kamu
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "RandomFlip": RandomFlip,
            "RandomRotation": RandomRotation,
            "RandomZoom": RandomZoom,
        }
    )
    return model

model = load_model()

# Label kelas
class_names = ["Kucing 🐱", "Anjing 🐶"]

# --- UI ---
st.set_page_config(page_title="Klasifikasi Anjing vs Kucing 🐾", page_icon="🐶")
st.title("🐾 Klasifikasi Anjing vs Kucing")
st.write("Upload gambar, lalu model akan menebak apakah itu **Kucing** atau **Anjing**.")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # --- Preprocessing ---
    img_height, img_width = 180, 180   # ganti sesuai input model saat training
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.image.resize(img_array, [img_height, img_width])
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalisasi

    # Prediksi
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.subheader("📊 Hasil Prediksi")
    st.write(
        f"👉 Gambar ini kemungkinan besar adalah **{class_names[np.argmax(score)]}** "
        f"dengan tingkat keyakinan **{100 * np.max(score):.2f}%**"
    )
