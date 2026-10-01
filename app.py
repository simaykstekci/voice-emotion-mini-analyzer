import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import whisper
from transformers import pipeline


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Voice Emotion Analyzer",
    page_icon="🎙️",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎙️ Voice Emotion Analyzer")

st.write(
    "Analyze speech characteristics, detect vocal emotion "
    "and transcribe speech using AI."
)


# --------------------------------------------------
# Load Whisper
# --------------------------------------------------

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")


# --------------------------------------------------
# Load Emotion Recognition Model
# --------------------------------------------------

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "audio-classification",
        model="Dpngtm/wav2vec2-emotion-recognition"
    )


# --------------------------------------------------
# Upload Audio
# --------------------------------------------------

audio_file = st.file_uploader(
    "Upload a voice recording",
    type=["wav", "mp3", "ogg", "m4a"]
)


# --------------------------------------------------
# Main Analysis
# --------------------------------------------------

if audio_file is not None:

    # Play audio
    st.audio(audio_file)

    # Save temporary audio file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Load audio
    y, sr = librosa.load(
        "temp_audio.wav",
        sr=None
    )

    # --------------------------------------------------
    # Basic Audio Features
    # --------------------------------------------------

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    rms = float(
        np.mean(
            librosa.feature.rms(y=y)
        )
    )

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(y)
        )
    )

    st.subheader("📊 Voice Features")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Duration",
        f"{duration:.2f} sec"
    )

    col2.metric(
        "Energy",
        f"{rms:.4f}"
    )

    col3.metric(
        "Zero Crossing Rate",
        f"{zcr:.4f}"
    )


    # --------------------------------------------------
    # Simple Voice Tone Estimation
    # --------------------------------------------------

    if rms > 0.08:
        tone = "Energetic / Excited 🔥"

    elif rms < 0.03:
        tone = "Calm / Soft 😌"

    else:
        tone = "Neutral 🙂"

    st.subheader("🎯 Estimated Voice Tone")

    st.success(tone)

    st.caption(
        "This is a simple rule-based estimation based on "
        "audio energy."
    )


    # --------------------------------------------------
    # AI Emotion Recognition
    # --------------------------------------------------

    st.subheader("🤖 AI Emotion Recognition")

    try:

        emotion_model = load_emotion_model()

        with st.spinner(
            "AI is analyzing the emotional characteristics..."
        ):

            emotions = emotion_model(
                "temp_audio.wav"
            )

        top_emotion = emotions[0]

        emotion_name = top_emotion["label"]
        confidence = top_emotion["score"] * 100

        st.success(
            f"Detected Emotion: {emotion_name}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )

        # Show all detected emotions
        st.write("Emotion probabilities:")

        for emotion in emotions:

            label = emotion["label"]
            score = emotion["score"] * 100

            st.progress(
                float(emotion["score"]),
                text=f"{label}: {score:.1f}%"
            )

    except Exception as e:

        st.warning(
            "Emotion recognition model could not be loaded."
        )

        st.code(str(e))


    # --------------------------------------------------
    # Whisper Speech Transcription
    # --------------------------------------------------

    st.subheader("📝 Speech Transcription")

    with st.spinner(
        "Whisper is transcribing the audio..."
    ):

        whisper_model = load_whisper()

        result = whisper_model.transcribe(
            "temp_audio.wav"
        )

    st.info(
        result["text"]
    )


    # --------------------------------------------------
    # Waveform
    # --------------------------------------------------

    st.subheader("🌊 Waveform")

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    librosa.display.waveshow(
        y,
        sr=sr,
        ax=ax
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.set_title(
        "Audio Waveform"
    )

    st.pyplot(fig)


    # --------------------------------------------------
    # MFCC
    # --------------------------------------------------

    st.subheader("🎵 MFCC Features")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 4)
    )

    img = librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        ax=ax2
    )

    ax2.set_title(
        "Mel-Frequency Cepstral Coefficients"
    )

    fig2.colorbar(
        img,
        ax=ax2
    )

    st.pyplot(fig2)


    # --------------------------------------------------
    # Spectrogram
    # --------------------------------------------------

    st.subheader("🔊 Spectrogram")

    spectrogram = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(y)
        ),
        ref=np.max
    )

    fig3, ax3 = plt.subplots(
        figsize=(10, 4)
    )

    img2 = librosa.display.specshow(
        spectrogram,
        x_axis="time",
        y_axis="hz",
        sr=sr,
        ax=ax3
    )

    ax3.set_title(
        "Audio Spectrogram"
    )

    fig3.colorbar(
        img2,
        ax=ax3
    )

    st.pyplot(fig3)


    # --------------------------------------------------
    # Footer
    # --------------------------------------------------

    st.caption(
        "Voice Emotion Analyzer — "
        "Speech analysis prototype using "
        "Python, Librosa, OpenAI Whisper "
        "and Wav2Vec2."
    )