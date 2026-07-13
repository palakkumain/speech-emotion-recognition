# 🎤 Speech Emotion Recognition using Machine Learning

A Machine Learning project that automatically detects human emotions from speech signals using audio feature extraction and the **RAVDESS Emotional Speech Dataset**.

The project extracts MFCC, Chroma, and Mel Spectrogram features from audio recordings and classifies emotions using a **Random Forest Classifier**. It also includes real-time emotion prediction from a microphone and a desktop GUI built with Tkinter.

---

## 📌 Features

- 🎙️ Emotion prediction from WAV audio files
- 🎤 Real-time microphone emotion detection
- 🖥️ Desktop GUI using Tkinter
- 🔊 Audio playback support
- 📊 Feature extraction using Librosa
- 🤖 Machine Learning model using Random Forest
- 💾 Pre-trained model included for quick testing

---

## 🧠 Machine Learning Workflow

1. Load audio dataset
2. Extract audio features
   - MFCC
   - Chroma Features
   - Mel Spectrogram
3. Create feature dataset
4. Train Random Forest classifier
5. Save trained model
6. Predict emotions from new speech recordings

---

## 🛠️ Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Tkinter
- Pygame
- SoundDevice
- Pickle

---

## 📂 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

Due to its size and licensing, the dataset is **not included** in this repository.

Download the dataset from the official RAVDESS source and place it in the following directory:

```
data/
└── ravdess-emotional-speech-audio/
```

---

## 📁 Project Structure

```
speech-emotion-recognition/
│
├── data/
│   └── ravdess-emotional-speech-audio/
│
├── models/
│   ├── emotion_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
│
├── gui_app.py
├── main.py
├── train_model.py
├── predict_emotion.py
├── realtime_emotion.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/palakkumain/speech-emotion-recognition.git
```

Move into the project folder:

```bash
cd speech-emotion-recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

Run:

```bash
python main.py
```

or

```bash
python train_model.py
```

The trained model will be saved inside the **models** folder.

---

## 🎧 Predict Emotion from an Audio File

```bash
python predict_emotion.py
```

---

## 🎙️ Real-Time Emotion Prediction

```bash
python realtime_emotion.py
```

Speak into your microphone and the model will predict your emotion.

---

## 🖥️ Launch the GUI

```bash
python gui_app.py
```

The GUI allows users to:

- Browse WAV files
- Play audio
- Detect emotions
- Display prediction results

---

## 📊 Model

**Algorithm:** Random Forest Classifier

**Feature Extraction:**

- MFCC
- Chroma Features
- Mel Spectrogram

---

## 🔮 Future Improvements

- Deep Learning (CNN + LSTM)
- Transformer-based Speech Models
- Streamlit Web Application
- Support for additional datasets
- Improved feature engineering
- Model deployment using Flask/FastAPI

---

## 👩‍💻 Author

**Palak Kumain**

BCA (Artificial Intelligence & Data Science)

Graphic Era Hill University

GitHub: https://github.com/palakkumain

---

## ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.