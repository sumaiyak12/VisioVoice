import streamlit as st
from deep_translator import GoogleTranslator
import pyttsx3
import tempfile
import os

# ========== APP MAIN FUNCTION ==========
def main():
    st.set_page_config(page_title="VisioVoice - Translator & TTS", layout="centered")
    st.title("🎧 VisioVoice – Translate & Speak Text")

    st.write("Enter text below, choose a language, and hear it spoken aloud!")

    # ----- INPUT -----
    text_input = st.text_area("✏️ Enter text here:", "")
    language = st.selectbox(
        "🌐 Select output language:",
        ["english", "hindi", "french", "spanish", "german", "arabic", "japanese"]
    )

    if st.button("Translate & Speak"):
        if text_input.strip() == "":
            st.warning("Please enter some text first!")
            return

        try:
            # ----- TRANSLATION -----
            translated = GoogleTranslator(source="auto", target=language).translate(text_input)
            st.success(f"**Translated Text ({language.capitalize()}):**")
            st.write(translated)

            # ----- TEXT TO SPEECH -----
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                engine.save_to_file(translated, tmpfile.name)
                engine.runAndWait()
                tmp_path = tmpfile.name

            # Play the audio
            audio_file = open(tmp_path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

            # Cleanup
            audio_file.close()
            os.remove(tmp_path)

        except Exception as e:
            st.error(f"Error: {str(e)}")


# ========== RUN APP ==========
if __name__ == "__main__":
    main()
