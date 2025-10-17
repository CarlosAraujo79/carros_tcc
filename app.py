import streamlit as st
from PIL import Image
import tempfile
from plate_utils import detect_plate_and_text

st.title("🔍 Detector de Placas de Carro")

uploaded_file = st.file_uploader("Envie uma imagem com uma placa de carro", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Imagem enviada", use_container_width=True)

    with st.spinner("Detectando placa e texto..."):
        # Salvar imagem temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            result_img, plate_text = detect_plate_and_text(tmp.name)

        st.image(result_img, caption="✅ Placa detectada", use_column_width=True)

        if plate_text:
            st.success(f"**Texto detectado:** `{plate_text}`")
        else:
            st.warning("Nenhum texto de placa foi detectado.")
