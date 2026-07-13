import sounddevice as sd
import librosa
import numpy as np
import pickle
import tempfile
import scipy.io.wavfile as wav

#  Load model, encoder, and scaler
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def extract_features(file_path):
    """Extract MFCC, Chroma, and Mel features"""
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

def predict_emotion_from_file(file_path):
    """Predict emotion from .wav file"""
    features = extract_features(file_path)
    features = scaler.transform([features])
    prediction = model.predict(features)
    return encoder.inverse_transform(prediction)[0]

def record_audio(duration=3, fs=22050):
    """Record real-time audio"""
    print(" Speak now... (Recording for 3 seconds)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print(" Recording complete.")
    
    # Save temp wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wav.write(temp_file.name, fs, (audio * 32767).astype(np.int16))
        return temp_file.name

#  Record and predict
audio_path = record_audio()
predicted_emotion = predict_emotion_from_file(audio_path)
print(f" Predicted Emotion: {predicted_emotion}")
