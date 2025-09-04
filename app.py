import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    repo_id = "zahratalitha/klasifikasikucing"  
    filename = "kucinganjing_full.h5"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()
class_names = ["Kucing 🐱", "Anjing 🐶"]

st.title("🐶🐱 Klasifikasi Gambar: Anjing vs Kucing")

uploaded_file = st.file_uploader("Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert ke grayscale karena model input (224,224,1)
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # (1,224,224,1)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.subheader("📊 Hasil Prediksi")
    st.write(
        f"👉 Gambar ini kemungkinan besar adalah **{class_names[np.argmax(score)]}** "
        f"dengan tingkat keyakinan **{100 * np.max(score):.2f}%**"
    )
