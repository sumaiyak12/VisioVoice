import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from io import BytesIO
from PIL import Image
import base64

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="VisioVoice", page_icon="🧠", layout="wide")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background: linear-gradient(to right, #f8fafc, #eef2f3);
            color: #000;
            font-family: 'Segoe UI', sans-serif;
        }
            div[data-testid="stAlert"] p {
            color: #000000 !important;
        }
            /*  Make the text inside the yellow alert banner black */
            div[data-testid="stAlert"] {
            color: #000000;
            background-color: #fff9c4 !important; /* soft yellow */
            font-weight: 600;
            border-radius: 8px;
            border: 1px solid #f0e68c;
        }

        /*  Make the "Browse files" button text white */
            button[kind="secondary"] {
            color: white !important;
            background-color: #111827 !important;
            border-radius: 8px;
            font-weight: 600;
        }
            button[kind="secondary"]:hover {
            background-color: #1f2937 !important;
        }
        /* Title Styling */
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            color: #111827;
            margin-top: -20px;
            animation: fadeInDown 1.5s ease;
        }
        @keyframes fadeInDown {
            from {opacity: 0; transform: translateY(-20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        /* Subtext */
        .subtitle {
            text-align: center;
            color: #374151;
            margin-bottom: 20px;
            font-size: 1rem;
            animation: fadeIn 2s ease;
        }
        /* Upload Box */
        .uploadedFile {color: #000 !important;}
        div[data-testid="stFileUploader"] > label {
            font-weight: 600;
            color: #000;
        }
        /* Result Card */
        .result-card {
            background: #fff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-top: 25px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .result-card:hover {
            transform: scale(1.01);
            box-shadow: 0 6px 25px rgba(0,0,0,0.12);
        }
        .desc-text {
            color: #000;
            font-size: 1.1rem;
            line-height: 1.6;
        }
           div[data-testid="stNotification"] {
            color: black !important;
        background-color: #fff9c4 !important;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #f0e68c;
        }

        button[kind="secondary"] {
        color: white !important;
        background-color: #111827 !important;
        border-radius: 8px;
        font-weight: 600;
        }
        button[kind="secondary"]:hover {
        background-color: #1f2937 !important;
        }
        /* Buttons */
        .stButton button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1d4ed8;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- TITLE ----------------------
st.markdown("<h1 class='main-title'>VisioVoice: Smart Image Description</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image to generate a detailed description — listen to it, translate it, and more!</p>", unsafe_allow_html=True)

# ---------------------- FILE UPLOAD ----------------------
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("✨ Processing your image... please wait"):
        try:
            # Use BLIP model for image captioning
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

            inputs = processor(images=image, return_tensors="pt")
            output = model.generate(**inputs)
            description = processor.decode(output[0], skip_special_tokens=True)

            # --------------- RESULT SECTION ---------------
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"<p class='desc-text'><b>📝 Description:</b> {description}</p>", unsafe_allow_html=True)

            # Convert text to audio
            tts = gTTS(description)
            audio = BytesIO()
            tts.write_to_fp(audio)
            st.audio(audio.getvalue(), format='audio/mp3')

            # Translation feature (optional)
            st.markdown("**🌐 Translate Description:**")
            lang = st.selectbox("Choose language", ["English", "Hindi", "French", "Spanish", "German"])
            if lang != "English":
                translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{lang[:2].lower()}")
                translated_text = translator(description)[0]['translation_text']
                st.markdown(f"**{lang} Translation:** {translated_text}")

            # Copy and download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download Description", description, file_name="visio_description.txt")
            with col2:
                st.code(description, language='text')

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

else:
    st.warning(" Please upload an image to start.")
