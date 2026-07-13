import librosa
import numpy as np
import pickle

# Load the saved model, label encoder, and scaler
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def extract_features(file_path):
    """Extract MFCC, Chroma, and Mel features from audio file"""
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

def predict_emotion(file_path):
    """Predict emotion from an audio file"""
    features = extract_features(file_path)
    features = scaler.transform([features])
    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)[0]
    return emotion

# Test with a new RAVDESS audio file
test_file = r"C:\Users\palak\OneDrive\Desktop\SpeechEmotionRecognition\ravdess-emotional-speech-audio\Actor_01\03-01-02-01-01-01-01.wav"
  # Change path as needed
predicted_emotion = predict_emotion(test_file)
print(f"Predicted Emotion: {predicted_emotion}")
