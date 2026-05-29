import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import os
import queue
import time
import urllib.request
from openai import OpenAI  # The xAI API is fully OpenAI-compatible

# ================== CUSTOM BRANDING & VIOLET STYLING ==================
st.set_page_config(
    page_title="GlobalInternet.py — SignBridge AI x Grok",
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
    .chat-box { background-color: #160f2e; border: 1px solid #3a1f5d; padding: 15px; border-radius: 8px; min-height: 250px; max-height: 400px; overflow-y: auto; }
    </style>
""", unsafe_allow_html=True)

# Validate Grok API Key Existence securely
grok_key = st.secrets.get("XAI_API_KEY", os.environ.get("XAI_API_KEY", ""))

# Shared Thread-Safe Queue to bridge Camera Thread with UI Thread safely
if "sign_queue" not in st.session_state:
    st.session_state.sign_queue = queue.Queue()
if "live_chat_log" not in st.session_state:
    st.session_state.live_chat_log = []

# --- COMPUTER VISION WEIGHT DOWNLOAD ROUTE ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

@st.cache_resource
def download_ai_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Initializing SignBridge AI Core Weights... Please wait."):
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

try:
    local_model_file = download_ai_model()
    model_error_flag = False
except Exception as e:
    model_error_flag = True
    local_model_file = None

# ================== SIDEBAR CONFIGURATION LOGIC ==================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌐 GlobalInternet.py</div>', unsafe_allow_html=True)
    language = st.selectbox("🌐 Target Translation Array / Langue:", ["English", "Français", "Português"])
    
    st.markdown("<hr style='border-color: #3a1f5d;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #bf80ff;'>📷 Select Hardware Input</h3>", unsafe_allow_html=True)
    camera_facing = st.selectbox(
        "Camera Lens Array:",
        options=["user", "environment"],
        format_func=lambda x: "📱 Laptop Built-in Cam" if x == "user" else "📸 External Desktop USB"
    )
    
    st.markdown("<hr style='border-color: #3a1f5d;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #bf80ff;'>🛠️ Core Infrastructure Desk</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-info">
        <strong>Coder-in-Chief:</strong><br><span style="color: #fff; font-weight:600;">Gesner Deslandes</span><br><br>
        <strong>Engineering Support Line:</strong><br><span style="color: #bf80ff;">(509)-47385663</span><br><br>
        <strong>Engine Mode:</strong> <span style="color: #00ffca;">{'Grok AI Mesh Connected' if grok_key else 'Local Rule Engine'}</span>
    </div>
    """, unsafe_allow_html=True)

# ================== LOCALIZATION DICTIONARY ==================
strings = {
    "English": {"title": "SignBridge AI Real-Time Stream", "sub": "Live Grok-Powered Translation Telemetry Desk", "step1": "🎥 Video Stream", "step2": "💬 Live Grok Chat Logs", "clear": "Clear Logs"},
    "Français": {"title": "Flux en Temps Réel SignBridge AI", "sub": "Bureau de télémétrie Grok AI", "step1": "🎥 Pipeline Vidéo", "step2": "💬 Journaux de transcription", "clear": "Effacer"},
    "Português": {"title": "Fluxo em Tempo Real SignBridge AI", "sub": "Balcão de telemetria Grok AI", "step1": "🎥 Pipeline de Vídeo", "step2": "💬 Registros de transcrição", "clear": "Limpar"}
}
current_lang = strings[language]

st.markdown(f'<div class="main-header">{current_lang["title"]}</div>', unsafe_allow_html=True)
st.write(f"<i style='color:#aaa2bc;'>{current_lang['sub']}</i>", unsafe_allow_html=True)

# ================== MODERN MEDIAPIPE INITIALIZATION ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- COMPUTER VISION CORE PROCESSING ENGINE ---
class RealTimeSignTransformer(VideoProcessorBase):
    def __init__(self, result_queue, model_path, api_key, target_language):
        self.result_queue = result_queue
        self.target_language = target_language
        self.active_ai = False
        
        # Instantiate Grok API Client if the key configuration is verified
        if api_key:
            self.grok_client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        else:
            self.grok_client = None

        if model_path and os.path.exists(model_path):
            try:
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.6
                )
                self.landmarker = HandLandmarker.create_from_options(options)
                self.active_ai = True
            except:
                self.active_ai = False
        
        self.frame_counter = 0
        self.prev_frame_time = 0
        self.last_api_call_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape
        self.frame_counter += 1
        
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 30
        self.prev_frame_time = new_frame_time
        
        img = cv2.flip(img, 1) 
        telemetry_log = "Telemetry Mode: Thread Safe"
        
        if self.active_ai:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
            try:
                detection_result = self.landmarker.detect(mp_image)
                if detection_result and detection_result.hand_landmarks:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    
                    # Trace visual nodes
                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(img, (x, y), 5, (230, 30, 150), -1) 

                    # RATE LIMIT PROTECTION: Only call Grok once every 1.5 seconds if hands move significantly
                    current_time = time.time()
                    if current_time - self.last_api_call_time > 1.5 and self.grok_client:
                        self.last_api_call_time = current_time
                        
                        # Pack all 21 hand coordinate structures into a small JSON string payload
                        coords_summary = [{"id": i, "x": round(lm.x, 3), "y": round(lm.y, 3), "z": round(lm.z, 3)} for i, lm in enumerate(hand_landmarks)]
                        
                        # Query Grok to evaluate the telemetry packet
                        response = self.grok_client.chat.completions.create(
                            model="grok-4.3",
                            messages=[
                                {"role": "system", "content": f"You are an expert Sign Language translation model. Analyze the 21 hand landmark coordinates provided. Translate them into a single word or short phrase in {self.target_language}. Respond ONLY with the translated phrase, nothing else."},
                                {"role": "user", "content": f"Landmarks: {str(coords_summary)}"}
                            ],
                            max_tokens=10,
                            temperature=0.1
                        )
                        grok_result = response.choices[0].message.content.strip()
                        if grok_result:
                            self.result_queue.put(grok_result)
                            
                    telemetry_log = f"Grok Node Streaming Matrix: {width}x{height}"
            except Exception as e:
                telemetry_log = f"Grok API Error: {str(e)}"

        cv2.putText(img, f"FPS: {int(fps)} | Matrix: {width}x{height}", (20, height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, telemetry_log, (20, height - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- CAMERA REGISTRATION UI LAYOUT ---
col_video, col_chat = st.columns([1, 1])

with col_video:
    st.write(f"<h3 style='color: #bf80ff;'>{current_lang['step1']}</h3>", unsafe_allow_html=True)
    
    webrtc_streamer(
        key="realtime-bridge-v8", 
        video_processor_factory=lambda: RealTimeSignTransformer(
            st.session_state.sign_queue, 
            local_model_file,
            grok_key,
            language
        ),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": camera_facing}, "audio": False}
    )

# --- REAL-TIME LIVE TRANSCRIPT DESK PANEL ---
with col_chat:
    st.write(f"<h3 style='color: #bf80ff;'>{current_lang['step2']}</h3>", unsafe_allow_html=True)
    
    while not st.session_state.sign_queue.empty():
        try:
            new_sign = st.session_state.sign_queue.get_nowait()
            st.session_state.live_chat_log.append(new_sign)
        except queue.Empty:
            break

    chat_html = "<div class='chat-box'>"
    if st.session_state.live_chat_log:
        for idx, word in enumerate(st.session_state.live_chat_log):
            chat_html += f"<p style='margin:6px 0; font-family:monospace;'><span style='color:#00ffca;'>[Grok #{idx+1}]:</span> <strong style='color:#fff; font-size:1.1rem;'>{word}</strong></p>"
    else:
        chat_html += "<p style='color:#6a5f80; font-style:italic;'>Waiting for Grok AI mesh telemetry stream...</p>"
    chat_html += "</div>"
    
    st.markdown(chat_html, unsafe_allow_html=True)
    
    st.write("##")
    if st.button(current_lang["clear"], use_container_width=True):
        st.session_state.live_chat_log = []
        st.rerun()

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
