import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
import pytesseract
import tempfile
import base64
from gtts import gTTS
import av

# === Load YOLO Models ===
@st.cache_resource
def load_models():
    indoor_model = YOLO("models/indoor.pt")
    outdoor_model = YOLO("models/outdoor.pt")
    return indoor_model, outdoor_model

indoor_model, outdoor_model = load_models()

OUTDOOR_CLASS_NAMES = [
    'Ambulance', 'Auto-Rikshaw', 'bike', 'bus', 'car',
    'puddle', 'stairs', 'truck', 'van', 'zebra-crossing'
]

# === OCR Preprocessing Function ===
def preprocess_image(pil_image):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    pil_thresh = Image.fromarray(thresh)
    enhancer = ImageEnhance.Contrast(pil_thresh)
    enhanced_image = enhancer.enhance(2.0)
    return enhanced_image

# === Speak Text via gTTS ===
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        tmpfile.seek(0)
        audio_bytes = tmpfile.read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# === Streamlit UI ===
st.set_page_config(page_title="Assistive App", layout="wide")
st.title("🧠 Assistive Vision System")

app_mode = st.sidebar.selectbox("Choose Mode", ["🧍 Object Detection", "🔠 OCR to TTS"])

# === OBJECT DETECTION MODE ===
if app_mode == "🧍 Object Detection":
    env = st.radio("Select Environment:", ["Indoor", "Outdoor"])
    model = indoor_model if env == "Indoor" else outdoor_model
    class_names = model.names if env == "Indoor" else OUTDOOR_CLASS_NAMES

    st.info(f"Running real-time object detection using the {env.lower()} model")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_sentence = ""

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(source=img, conf=0.4)[0]
            labels = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                try:
                    label = class_names[cls_id]
                except IndexError:
                    continue
                labels.append(label)
                xyxy = box.xyxy[0].int().tolist()
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if labels:
                sentence = " and ".join(set(labels)) + " ahead"
                if sentence != self.last_sentence:
                    self.last_sentence = sentence
                    speak_text(sentence)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="object-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration = {
            
            "iceServers": [
                {
                    "urls": "stun:stun.l.google.com:19302"
                },
                {
                    "urls": ["turn:global.turn.twilio.com:3478?transport=udp"],
                    "username": "b9e6f8ff9be8b7303e3520570113cff848385c3c60b83b17adaab2e5a607385c",
                    "credential": "ZT8h0y7ShKOWLmtyYH845iay2/w+0i0GNFVwZ73/1qw="
                }
            ]
        },
    )

# === OCR TO TTS MODE ===
elif app_mode == "🔠 OCR to TTS":
    st.subheader("📷 Capture Image for OCR")
    img_file = st.camera_input("Take a picture")

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Captured Image", use_column_width=True)

        processed = preprocess_image(image)
        st.image(processed, caption="Preprocessed Image", use_column_width=True)

        text = pytesseract.image_to_string(processed)
        text = " ".join(text.split())

        if text:
            st.subheader("📝 Extracted Text")
            st.success(text)
            speak_text(text)
        else:
            st.warning("No text found in the image.")
