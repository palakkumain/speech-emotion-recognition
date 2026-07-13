# 🎤 Speech Emotion Recognition

## Overview

This project detects human emotions from speech using Machine Learning and audio signal processing.

The model is trained on the RAVDESS emotional speech dataset using Librosa for feature extraction and a Random Forest classifier for emotion prediction.

---

## Features

- Emotion prediction from WAV files
- Real-time microphone emotion detection
- Desktop GUI using Tkinter
- Audio playback support
- Machine Learning model trained on RAVDESS dataset

---

## Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Tkinter
- Pygame
- SoundDevice

---

## Dataset

This project uses the **RAVDESS Emotional Speech Dataset**.

The dataset is not included in this repository because of GitHub size limits.

Download it from the official source and place it inside:

data/ravdess-emotional-speech-audio/

---

## Project Structure

SpeechEmotionRecognition/

├── data/

├── models/

├── gui_app.py

├── train_model.py

├── predict_emotion.py

├── realtime_emotion.py

├── main.py

├── README.md

├── requirements.txt

└── .gitignore

---

## How to Train

Run:

python main.py

or

python train_model.py

---

## How to Predict

python predict_emotion.py

---

## Real-Time Prediction

python realtime_emotion.py

---

## GUI

python gui_app.py

---

## Future Improvements

- Deep Learning using CNN/LSTM
- Streamlit Web Application
- More datasets
- Better feature engineering

---

## Author

Palak Kumain