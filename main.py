import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

#  Folder name
DATA_PATH = "data/ravdess-emotional-speech-audio"

# Emotion mapping
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    """Extract MFCC, Chroma, and Mel features"""
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

features = []
labels = []

#  Extract features from dataset
for actor_folder in os.listdir(DATA_PATH):
    actor_path = os.path.join(DATA_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for file_name in os.listdir(actor_path):
        if file_name.endswith(".wav"):
            emotion_code = file_name.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            file_path = os.path.join(actor_path, file_name)
            data = extract_features(file_path)
            features.append(data)
            labels.append(emotion)

#  Create DataFrame
df = pd.DataFrame(features)
df['emotion'] = labels
df.to_csv("features.csv", index=False)
print(" Features extracted and saved to features.csv")

#  Split data
X = df.drop(columns=['emotion'])
y = df['emotion']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

#  Check accuracy
acc = model.score(X_test, y_test)
print(f" Model trained successfully with accuracy: {acc:.2f}")

#  Save model, encoder, and scaler
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model, encoder, and scaler saved successfully!") 