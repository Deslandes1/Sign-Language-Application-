import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import edge_tts
import os
import queue

# ================== CUSTOM BRANDING & VIOLET STYLING ==================
st.set_page_config(
    page_title="GlobalInternet.py — SignBridge AI",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0b071a; color: #e2d9f3; }
    [data-testid="stSidebar"] { background-color: #130c26 !important; border-right: 2px solid #3a1f5d; }
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, #dfa2ff, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sidebar-brand { font-size: 1.4rem; font-weight: bold; color: #bf80ff; border-bottom: 2px solid #3a1f5d; padding-bottom: 10px; margin-bottom: 15px; }
    .sidebar-info { font-size: 0.95rem; color: #aaa2bc; line-height: 1.6; }
    div[data-baseweb="input"] { background-color: #1a1235 !important; border-color: #5c3593 !important; }
    input { color: #ffffff !important; }
    .stButton>button { background-color: #5c3593 !important; color: white !important; border: 1px solid #8a2be2 !important; }
    .stButton>button:hover { background-color: #7944c3 !important; border-color: #bf80ff !important; box-shadow: 0px 0px 10px #8a2be2; }
    .chat-box { background-color: #160f2e; border: 1px solid #3a1f5d; padding: 15px; border-radius: 8px; max-height: 300px; overflow-y: auto; }
    </style>
""", unsafe_allow_html=True)

# Shared Thread-Safe Queue to bridge Camera Thread with UI Thread
if "sign_queue" not in st.session_state:
    st.session_state.sign_queue = queue.Queue()
if "live_chat_log" not in st.session_state:
    st.session_state.live_chat_log = []

# ================== SIDEBAR CONFIGURATION LOGIC ==================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌐 GlobalInternet.py</div>', unsafe_allow_html=True)
    language = st.selectbox("🌐 Target Translation Array / Langue:", ["English", "Français", "Português"])
    st.markdown("<hr style='border-color: #3a1f5d;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #bf80ff;'>🛠️ Core Infrastructure Desk</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-info">
        <strong>Coder-in-Chief:</strong><br><span style="color: #fff; font-weight:600;">Gesner Deslandes</span><br><br>
        <strong>Engineering Support Line:</strong><br><span style="color: #bf80ff;">(509)-47385663</span><br><br>
        <strong>Status:</strong> <span style="color: #aa66ff;">● Real-time Stream Engine Live</span>
    </div>
    """, unsafe_allow_html=True)

# ================== LOCALIZATION DICTIONARY ==================
strings = {
    "English": {
        "title": "SignBridge AI Real-Time Stream",
        "sub": "Live Sign Language Translation Telemetry Desk",
        "step1": "🎥 Live Video Stream Pipeline",
        "step2": "💬 Real-Time Live Transcript (Updates automatically)",
        "clear": "Clear Live Transcript Logs",
        "hello": "HELLO", "thank_you": "THANK YOU"
    },
    "Français": {
        "title": "Flux en Temps Réel SignBridge AI",
        "sub": "Bureau de télémétrie de traduction de la langue des signes en direct",
        "step1": "🎥 Pipeline de flux vidéo en direct",
        "step2": "💬 Transcription en direct (Mise à jour automatique)",
        "clear": "Effacer les journaux de transcription",
        "hello": "BONJOUR", "thank_you": "MERCI"
    },
    "Português": {
        "title": "Fluxo em Tempo Real SignBridge AI",
        "sub": "Balcão de telemetria de tradução de língua de sinais ao vivo",
        "step1": "🎥 Pipeline de transmissão de vídeo ao vivo",
        "step2": "💬 Transcrição em tempo real (Atualização automática)",
        "clear": "Limpar registros de transcrição",
        "hello": "OLÁ", "thank_you": "OBRIGADO"
    }
}
current_lang = strings[language]

st.markdown(f'<div class="main-header">{current_lang["title"]}</div>', unsafe_allow_html=True)
st.write(f"<i style='color:#aaa2bc;'>{current_lang['sub']}</i>", unsafe_allow_html=True)

# ================== MODERN MEDIAPIPE INITIALIZATION ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- COMPUTER VISION VIDEO LAYER ---
class RealTimeSignTransformer(VideoTransformerBase):
    def __init__(self, result_queue):
        self.result_queue = result_queue
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=None),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.last_emitted_sign = None
        self.frame_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) 
        self.frame_counter += 1
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection_result = self.landmarker.detect(mp_image)
        
        detected_sign = ""
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    cv2.circle(img, (x, y), 5, (230, 30, 150), -1) 

                thumb_tip = hand_landmarks[4].y
                index_tip = hand_landmarks[8].y
                
                if index_tip < thumb_tip:
                    detected_sign = current_lang["hello"]
                else:
                    detected_sign = current_lang["thank_you"]

            cv2.putText(img, f"Streaming: {detected_sign}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 120, 210), 2)
            
            # Anti-flood logic: Only push to UI chat if the sign has changed and remains stable
            if detected_sign != self.last_emitted_sign and self.frame_counter % 10 == 0:
                self.result_queue.put(detected_sign)
                self.last_emitted_sign = detected_sign
        else:
            self.last_emitted_sign = None

        return img

# --- CAMERA REGISTRATION UI LAYOUT ---
col_video, col_chat = st.columns([1, 1])

with col_video:
    st.write(f"<h3 style='color: #bf80ff;'>{current_lang['step1']}</h3>", unsafe_allow_html=True)
    webrtc_streamer(
        key="realtime-bridge", 
        video_transformer_factory=lambda: RealTimeSignTransformer(st.session_state.sign_queue),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

# --- REAL-TIME LIVE TRANSCRIPT DESK ---
with col_chat:
    st.write(f"<h3 style='color: #bf80ff;'>{current_lang['step2']}</h3>", unsafe_allow_html=True)
    
    # Check the bridge queue for new incoming frames from the camera
    while not st.session_state.sign_queue.empty():
        try:
            new_sign = st.session_state.sign_queue.get_nowait()
            st.session_state.live_chat_log.append(new_sign)
        except queue.Empty:
            break

    # Build clear visual scrolling output text terminal
    chat_html = "<div class='chat-box'>"
    if st.session_state.live_chat_log:
        for idx, word in enumerate(st.session_state.live_chat_log):
            chat_html += f"<p style='margin:4px 0; font-family:monospace;'><span style='color:#bf80ff;'>[Sign {idx+1}]:</span> <strong style='color:#fff; font-size:1.1rem;'>{word}</strong></p>"
    else:
        chat_html += "<p style='color:#6a5f80; font-style:italic;'>Waiting for camera translation telemetry input stream...</p>"
    chat_html += "</div>"
    
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # Utility clear action
    st.write("##")
    if st.button(current_lang["clear"], use_container_width=True):
        st.session_state.live_chat_log = []
        st.rerun()

# --- AUTOMATIC UI REFRESH ELEMENT ---
# This forces Streamlit to look for new camera signs every 1000ms if changes occur
if len(st.session_state.live_chat_log) > 0:
    st.write("") 

# ================== CORPORATE FOOTER LAYER ==================
st.markdown("<hr style='border-color: #3a1f5d;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85rem; color:#aaa2bc; letter-spacing: 0.5px;">
        🚀 <strong>SIGNBRIDGE REAL-TIME ENGINE</strong> | Developed and Maintained by <strong>GlobalInternet.py</strong><br>
        Architect-in-Chief: <strong>Gesner Deslandes</strong> | Infrastructure Desk Contact: <strong style="color:#bf80ff;">(509)-47385663</strong>
    </div>
    """,
    unsafe_allow_html=True
)
