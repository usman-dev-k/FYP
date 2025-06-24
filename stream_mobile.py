import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
from TTS.api import TTS
import pytesseract
import tempfile
import base64

# ========== Load Models ==========
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

INDOOR_MODEL_PATH = "/home/sag_umt/yolo_project/yolov8s.pt"
OUTDOOR_MODEL_PATH = "/home/sag_umt/Music/outdoor_best_weights/weights/best.pt"

indoor_model = YOLO(INDOOR_MODEL_PATH)
outdoor_model = YOLO(OUTDOOR_MODEL_PATH)

OUTDOOR_CLASS_NAMES = [
    'Ambulance', 'Auto-Rikshaw', 'bike', 'bus', 'car',
    'puddle', 'stairs', 'truck', 'van', 'zebra-crossing'
]

def preprocess_image(pil_image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    # Resize to improve OCR on small text
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Optional: Increase contrast
    pil_thresh = Image.fromarray(thresh)
    enhancer = ImageEnhance.Contrast(pil_thresh)
    enhanced_image = enhancer.enhance(2.0)

    return enhanced_image



# ========== Streamlit UI ==========
st.set_page_config(page_title="Real-Time Assistive App", layout="wide")
st.title("🧠 Assistive Vision System")

app_mode = st.sidebar.selectbox("Choose Mode", ["🧍 Object Detection", "🔠 OCR to TTS"])

# ========== OBJECT DETECTION ==========
if app_mode == "🧍 Object Detection":
    env = st.radio("Select Environment:", ["Indoor", "Outdoor"])
    selected_model = indoor_model if env == "Indoor" else outdoor_model
    class_names = selected_model.names if env == "Indoor" else OUTDOOR_CLASS_NAMES

    st.info(f"Running real-time object detection using the {env.lower()} model")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_sentence = ""

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = selected_model.predict(source=img, conf=0.4)[0]

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
                cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if labels:
                sentence = " and ".join(set(labels)) + " ahead"
                if sentence != self.last_sentence:
                    self.last_sentence = sentence
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                        tts_model.tts_to_file(text=sentence, file_path=tmpfile.name)
                        tmpfile.seek(0)
                        audio_bytes = tmpfile.read()

                    b64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f"""
                    <audio autoplay>
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="object-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )


# ========== OCR TO TTS ==========
elif app_mode == "🔠 OCR to TTS":
    st.subheader("📷 Capture Image for OCR")

    captured_img = st.camera_input("Take a picture")

    if captured_img is not None:
        image = Image.open(captured_img)
        st.image(image, caption="Captured Image", use_column_width=True)
        
        preprocessed_img = preprocess_image(image)
        st.image(preprocessed_img, caption="Preprocessed Image", use_column_width=True)
        
        text = pytesseract.image_to_string(preprocessed_img)
        text = " ".join(text.split())

        if text:
            st.subheader("📝 Extracted Text")
            st.success(text)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tts_model.tts_to_file(text=text, file_path=tmpfile.name)
                tmpfile.seek(0)
                audio_bytes = tmpfile.read()

            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
            st.subheader("🗣 Speaking...")
            st.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.warning("No text found in the image.")

