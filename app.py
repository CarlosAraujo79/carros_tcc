import streamlit as st
from PIL import Image
import torch
from utils import detect_plate_and_text
import tempfile

st.title("Detector de Placas de Carro")

uploaded_file = st.file_uploader("Envie uma imagem com uma placa de carro", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    with st.spinner("Detectando placa e texto..."):
        # Salvar temporariamente a imagem
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            image.save(tmp.name)
            result_img, plate_text = detect_plate_and_text(tmp.name)

        st.image(result_img, caption="Placa detectada", use_column_width=True)
        st.success(f"Texto na placa: `{plate_text}`")
