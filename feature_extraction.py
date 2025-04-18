import librosa
import numpy as np
import re
import whisper

# Load Whisper model once
model = whisper.load_model("large")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # 1. Pitch
    pitch = librosa.yin(y, fmin=50, fmax=500)
    pitch_mean = float(np.mean(pitch)) if pitch.size > 0 else 0.0
    pitch_std = float(np.std(pitch)) if pitch.size > 0 else 0.0

    # 2. Jitter (basic proxy)
    jitter = float(np.mean(np.abs(np.diff(y)))) if y.size > 1 else 0.0

    # 3. Shimmer (basic proxy)
    shimmer = float(np.std(y)) if y.size > 0 else 0.0

    # 4. Duration
    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    # 5. Speech rate (beats per second)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    speech_rate = float(tempo) / 60.0

    # 6–10. Transcription-based features
    result = model.transcribe(audio_path, fp16=False)
    text = result["text"].strip()

    if not text:  # Ensure non-empty transcription
        return np.zeros((1, 10))

    words = re.findall(r'\b\w+\b', text)
    word_count = int(len(words))

    sentences = re.split(r'[.!?]+', text)
    sentence_count = int(len([s for s in sentences if len(s.strip()) > 0]))

    avg_words_per_sentence = float(word_count) / sentence_count if sentence_count > 0 else 0.0

    hesitations = int(len(re.findall(r'\b(uh+|um+)\b', text.lower())))

    # Now all features are scalar
    features = np.array([[pitch_mean, jitter, word_count, sentence_count, duration_sec, 
                          pitch_std, speech_rate, avg_words_per_sentence, hesitations, shimmer]])

    return features
