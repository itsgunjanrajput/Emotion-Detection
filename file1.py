import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from fer import FER
import numpy as np

# Title
st.title("😊 Real-Time Emotion Detection App")
st.markdown("Detect emotions from webcam in real-time using OpenCV and FER")

# Load emotion detector
emotion_detector = FER(mtcnn=True)

# Video transformer for streamlit-webrtc
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Detect emotions
        result = emotion_detector.detect_emotions(img)
        
        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            # Get the dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            # Draw box and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, dominant_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return img

# Start webcam
webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionTransformer)
