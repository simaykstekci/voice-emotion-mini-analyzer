# 🎙️ Voice Emotion Analyzer

### 🚀 Live Demo

👉 **[Try the Voice Emotion Analyzer](https://voice-emotion-mini-analyzer-3tfpatzcoujzhsbgv4j3y4.streamlit.app/)**

A Python-based speech analysis application that uses **Librosa, OpenAI Whisper and Wav2Vec2** to analyze voice characteristics, recognize emotional patterns and transcribe speech.

## 🚀 Project Overview

**Voice Emotion Analyzer** is a speech and audio analysis application developed to explore **audio signal processing, feature extraction, speech recognition, and AI-based emotion recognition**.

The application allows users to upload a voice recording and analyze its acoustic characteristics, estimate vocal tone, recognize emotional patterns using a pretrained AI model, and transcribe speech using OpenAI Whisper.


# 🎙️ Voice Emotion Mini Analyzer

A Python-based voice analysis application that processes uploaded audio files and performs basic emotion/tonality analysis using audio signal features and machine learning techniques.

## 🚀 Project Overview

**Voice Emotion Mini Analyzer** is a small-scale Speech Emotion Recognition (SER) project developed to explore audio processing, feature extraction, and machine learning.

The application allows users to upload a voice recording and analyzes the audio to provide an estimated emotional tone through an interactive web interface.

The project was developed as a practical study of **Python, audio signal processing, feature extraction, and machine learning**.

## ✨ Features

* 🎙️ Upload an audio file through the web interface
* 🔊 Process and analyze the uploaded voice recording
* 📊 Extract audio/speech-related features
* 🤖 Perform basic emotion/tonality prediction
* 📈 Display analysis results through an interactive interface
* 💻 Simple and user-friendly Streamlit interface

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Librosa**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **OpenAI Whisper**

## 🧠 How It Works

The application follows a basic audio analysis pipeline:

```text
Audio Upload
     ↓
Audio Preprocessing
     ↓
Feature Extraction
     ↓
Speech / Audio Analysis
     ↓
Emotion / Tonality Prediction
     ↓
Results
```

The uploaded audio is processed to extract relevant characteristics from the speech signal. These features are then used by the analysis pipeline to generate an estimated emotional tone.

## 📂 Project Structure

```text
voice-emotion-mini-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
└── ...
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/voice-emotion-mini-analyzer.git
```

### 2. Navigate to the project directory

```bash
cd voice-emotion-mini-analyzer
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will then be available locally in your browser.

## 🎯 Project Purpose

This project was developed to gain hands-on experience in:

* Speech and audio processing
* Feature extraction from audio signals
* Machine learning-based classification
* Python-based application development
* Building interactive data/AI applications with Streamlit

It also serves as a practical introduction to **Speech Emotion Recognition (SER)** and AI-based audio analysis.

## ⚠️ Disclaimer

The emotion analysis provided by this application is an **experimental estimate** and should not be interpreted as a reliable psychological or clinical assessment.

The accuracy of the results can vary depending on factors such as recording quality, background noise, speaker characteristics, language, and the training data used.

## 🔮 Future Improvements

Possible future improvements include:

* Training the model on a larger and more diverse speech emotion dataset
* Improving emotion classification accuracy
* Supporting additional emotion categories
* Adding visualization of extracted audio features
* Improving robustness against background noise
* Supporting real-time microphone input
* Deploying the application as a publicly accessible web demo

## 👩‍💻 Author

**Simay**

Software Engineering Student

---

⭐ This project was developed as a learning project focused on **Artificial Intelligence, Speech Processing, and Machine Learning**.
