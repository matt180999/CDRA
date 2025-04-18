# 🧠 Voice-Based Cognitive Decline Detection App

This Streamlit app analyzes audio input to assist in detecting potential signs of cognitive decline using speech patterns and acoustic features.

---

## 🚀 Features

- Upload audio samples (`.wav` recommended)
- Extract features from speech using `librosa` and `pydub`
- Transcribe using OpenAI's Whisper
- Predict cognitive status using a pre-trained ML model

---

## 🛠️ Technologies Used

- Python
- Streamlit
- NumPy, Pandas
- joblib (for loading the saved model)
- Whisper (for transcription)
- Librosa & Pydub (for audio processing)

---

## 🧪 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
