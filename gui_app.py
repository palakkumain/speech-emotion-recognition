import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import pickle
import pygame

# Load model, encoder, scaler
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

#  Initialize pygame for audio playback
pygame.mixer.init()

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

#  Prediction function
def predict_emotion(file_path):
    try:
        features = extract_features(file_path)
        features = scaler.transform([features])
        prediction = model.predict(features)
        emotion = encoder.inverse_transform(prediction)[0]
        return emotion
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        return None

#  Browse file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)

#  Play audio
def play_audio():
    file_path = entry_path.get()
    if not file_path:
        messagebox.showwarning("Warning", "Please select an audio file first!")
        return
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# Predict emotion
def detect_emotion():
    file_path = entry_path.get()
    if not file_path:
        messagebox.showwarning("Warning", "Please select an audio file first!")
        return
    emotion = predict_emotion(file_path)
    if emotion:
        lbl_result.config(text=f" Predicted Emotion: {emotion.upper()}", fg="green")

#  GUI setup
root = tk.Tk()
root.title("Speech Emotion Recognition")
root.geometry("500x300")
root.config(bg="#f2f2f2")

# UI Components
tk.Label(root, text="Select a .wav file to analyze emotion", font=("Helvetica", 12), bg="#f2f2f2").pack(pady=10)

frame = tk.Frame(root, bg="#f2f2f2")
frame.pack()

entry_path = tk.Entry(frame, width=40, font=("Helvetica", 10))
entry_path.grid(row=0, column=0, padx=5, pady=5)

btn_browse = tk.Button(frame, text="Browse", command=browse_file, bg="#4CAF50", fg="white", width=10)
btn_browse.grid(row=0, column=1, padx=5)

btn_play = tk.Button(root, text="▶ Play Audio", command=play_audio, bg="#2196F3", fg="white", width=20)
btn_play.pack(pady=10)

btn_detect = tk.Button(root, text=" Detect Emotion", command=detect_emotion, bg="#9C27B0", fg="white", width=20)
btn_detect.pack(pady=10)

lbl_result = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#f2f2f2")
lbl_result.pack(pady=10)

root.mainloop()
