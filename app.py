import streamlit as st
import os
import numpy as np
import joblib
from feature_extraction import extract_features
from pydub import AudioSegment
import tempfile
import whisper

# --- Page config ---
st.set_page_config(page_title="Cognitive Decline Risk Analysis Using Audio", layout="centered")

# --- Custom CSS styling (optional) ---
st.markdown("""
    <style>
        .main {
            background-color: #111827;
            color: white;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        .stButton>button {
            background-color: #6366f1;
            color: white;
        }
        .low-risk {
            color: #10b981;
            font-weight: bold;
        }
        .high-risk {
            color: #ef4444;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    return joblib.load("isolation_forest_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# Load Whisper model once
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("large")

whisper_model = load_whisper_model()

# --- UI Header ---
st.markdown("## 🧠Cognitive Decline Risk Analysis Using Audio")
st.markdown("Upload an audio file (.wav/.mp3)")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        # Convert MP3 to WAV if needed
        if uploaded_file.name.endswith(".mp3"):
            audio = AudioSegment.from_file(uploaded_file, format="mp3")
            audio.export(tmpfile.name, format="wav")
        else:
            tmpfile.write(uploaded_file.read())

        temp_path = tmpfile.name

    # Audio player
    st.audio(temp_path)

    # Extract features
    features = extract_features(temp_path)

    # Get prediction score
    raw_score = -model.decision_function(features)[0]

    # Normalize using pre-fitted scaler
    normalized_score = scaler.transform(np.array([[raw_score]]))[0][0]

    # Interpret result
    risk_level = "Low Risk" if normalized_score < 2.3 else "High Risk"
    risk_style = "low-risk" if normalized_score < 2.3 else "high-risk"

    # --- Transcription using Whisper ---
    result = whisper_model.transcribe(temp_path, fp16=False)
    transcription = result["text"].strip()

    st.markdown("### Transcription")
    if transcription:
        st.write(transcription)
    else:
        st.write("No speech detected in the audio.")

    # Prediction result
    st.markdown("### Prediction")
    st.markdown(f"**Risk Level:** <span class='{risk_style}'>{risk_level}</span>", unsafe_allow_html=True)
    

    # Clean up
    os.remove(temp_path)
