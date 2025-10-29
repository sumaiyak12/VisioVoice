# visio_voice_app.py

import streamlit as st
from transformers import pipeline
from gtts import gTTS
from io import BytesIO

# -----------------------------
# App Configuration
# -----------------------------
import streamlit as st
from PIL import Image
from gtts import gTTS
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------- Page Setup --------------------
st.set_page_config(page_title="VisioVoice", page_icon="🧠", layout="centered")

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
        /* General background and text color */
        body {
            background-color: #f8f9fa;
            color: #000000; /* text color black */
        }

        /* Center content properly */
        .block-container {
            max-width: 800px;
            margin: auto;
            padding-top: 2rem;
        }

        /* Title styling */
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            color: #000000; /* black text */
            text-align: center;
            margin-bottom: 0.3rem;
        }

        /* Subtitle styling */
        .sub-text {
            text-align: center;
            font-size: 1rem;
            color: #333333; /* darker gray for readability */
            margin-bottom: 2rem;
        }

        /* File uploader box */
        .uploadedFile {
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("<h1 class='main-title'>VisioVoice: Smart Image Description</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload an image to generate a clear, detailed description — and even listen to it!</p>", unsafe_allow_html=True)


# -----------------------------
# Image Upload Section
# -----------------------------
uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    st.info("Processing your image... please wait ⏳")

    # -----------------------------
    # Load image-to-text model
    # -----------------------------
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # Generate description
    result = captioner(uploaded_image)
    description = result[0]['generated_text']

    # Generate long, detailed version
    description_long = (
        "This image appears to depict " + description.lower() + ". "
        "The system observes various details like objects, environment, "
        "and overall composition to generate this description."
    )

    # Display description
    st.markdown("<div class='result-box'><b>📝 Generated Description:</b><br>" + description_long + "</div>", unsafe_allow_html=True)

    # -----------------------------
    # Text to Speech
    # -----------------------------
    tts = gTTS(description_long)
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    st.audio(audio_stream, format="audio/mp3")
else:
    st.warning("👆 Please upload an image to start.")
