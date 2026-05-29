import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import edge_tts
import os

# ================== MODERN MEDIAPIAPE INITIALIZATION ==================
# Using the updated non-deprecated vision tracking API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create drawing utilities for video overlay
mp_draw = mp.solutions.drawing_utils
mp_hands_connections = mp.solutions.hands_connections

st.title("🌐 SignBridge AI Chatbot — GlobalInternet.py")
st.subheader("Real-Time Sign Language Translation & Audio Sync")

# Initialize persistent session states for the chatbot conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "detected_text" not in st.session_state:
    st.session_state.detected_text = ""

# --- AUDIO GENERATION FUNCTION ---
def generate_audio(text, lang="en-US"):
    voice = "en-US-ChristopherNeural" if lang == "en-US" else "fr-FR-HenriNeural"
    output_file = "response.mp3"
    
    communicate = edge_tts.Communicate(text, voice)
    asyncio.run(communicate.save(output_file))
    return output_file

# --- COMPUTER VISION VIDEO LAYER ---
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        # Configure Hand Landmarker task options
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=None), # Uses native embedded models
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Natural mirror orientation
        
        # Convert frame format for processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        # Process landmarks with modern framework
        detection_result = self.landmarker.detect(mp_image)
        
        detected_sign = ""
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Map tracked skeleton onto video stream frame
                # Converting dict-style structural items to MediaPipe display structure
                for landmark in hand_landmarks:
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                
                # --- SIGN TRANSLATION MATRIX LOGIC ---
                thumb_tip = hand_landmarks[4].y
                index_tip = hand_landmarks[8].y
                if index_tip < thumb_tip:
                    detected_sign = "HELLO"
                else:
                    detected_sign = "THANK YOU"

            # Render overlay text on client screen
            cv2.putText(img, f"Sign: {detected_sign}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 199), 2)
            
            st.session_state.detected_text = detected_sign

        return img

# --- CAMERA INPUT FIELD ---
st.write("### 🎥 Step 1: Initialize Camera Stream")
webrtc_streamer(
    key="sign-streamer", 
    video_transformer_factory=SignLanguageTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

# --- CHATBOT INTERACTIVE INTERFACE ---
st.write("### 💬 Step 2: Live Translation Desk")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("Translated Text Input Vector:", value=st.session_state.detected_text)

with col2:
    st.write("##")
    send_btn = st.button("Send to Chatbot", use_container_width=True)

if send_btn and user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    
    bot_reply = f"System processed sign: '{user_input}'. How can I assist you further?"
    st.session_state.chat_history.append({"role": "bot", "text": bot_reply})
    
    audio_path = generate_audio(bot_reply)
    st.session_state.latest_audio = audio_path

# --- DISPLAY CONVERSATION STREAM ---
for msg in reversed(st.session_state.chat_history):
    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("assistant").write(msg["text"])
        if "latest_audio" in st.session_state and os.path.exists(st.session_state.latest_audio):
            st.audio(st.session_state.latest_audio, format="audio/mp3")

# ================== Developer Footer Core Layers ==================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; opacity: 0.8;">
        <strong>SIGNBRIDGE AI</strong> | Engineered by <strong>Gesner Deslandes</strong> (GlobalInternet.py)<br>
        📧 deslandes78@gmail.com | 📞 (509)-47385663
    </div>
    """,
    unsafe_allow_html=True
)
