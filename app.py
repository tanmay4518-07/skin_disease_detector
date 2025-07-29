import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
from reportlab.pdfgen import canvas

# === CONFIG ===
IMG_SIZE = 224
CLASS_NAMES = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
               'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("skin_cancer_model.h5")
    return model

model = load_model()

# === PREDICT FUNCTION ===
def predict(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_index = np.argmax(preds)
    top_class = CLASS_NAMES[top_index]
    confidence = float(preds[top_index])
    return top_class, confidence

# === PDF REPORT GENERATION ===
def generate_pdf(disease, confidence):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 14)
    c.drawString(100, 800, "Skin Cancer Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 770, f"Predicted Disease: {disease}")
    c.drawString(100, 750, f"Confidence: {confidence * 100:.2f}%")
    c.save()
    buffer.seek(0)
    return buffer

# === STREAMLIT UI ===
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔬 Skin Cancer Classifier")
st.markdown("Upload a skin image and this tool will predict the **most likely disease** among 7 skin cancer types.")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        predicted_disease, confidence = predict(image)

    st.markdown(f"### 🧾 Prediction: `{predicted_disease}`")
    st.progress(min(int(confidence * 100), 100), text=f"Confidence: {confidence * 100:.2f}%")

    if st.button("📄 Download PDF Report"):
        pdf_buffer = generate_pdf(predicted_disease, confidence)
        st.download_button(label="Download Report", data=pdf_buffer,
                           file_name="skin_cancer_report.pdf", mime="application/pdf")
