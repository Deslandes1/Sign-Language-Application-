import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import edge_tts
import os

# ================== CUSTOM BRANDING & VIOLET STYLING ==================
st.set_page_config(
    page_title="GlobalInternet.py — SignBridge AI",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS injection for full-scale deep purple and violet aesthetic
st.markdown("""
    <style>
    /* Main Background & Text Color Tweaks */
    .stApp {
        background-color: #0b071a;
        color: #e2d9f3;
    }
    
    /* Entire Sidebar Container Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #130c26 !important;
        border-right: 2px solid #3a1f5d;
    }
    
    /* Dynamic Neon Purple Header */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 800;
        background: linear-gradient(45deg, #b8860b, #9400d3, #da70d6);
        background: linear-gradient(45deg, #dfa2ff, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Accent Sidebar Styles */
    .sidebar-brand {
        font-size: 1.4rem;
        font-weight: bold;
        color: #bf80ff;
        border-bottom: 2px solid #3a1f5d;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .sidebar-info {
        font-size: 0.95rem;
        color: #aaa2bc;
        line-height: 1.6;
    }
    
    /* Style form boxes and text fields to fit the palette */
    div[data-baseweb="input"] {
        background-color: #1a1235 !important;
        border-color: #5c3593 !important;
    }
    input {
        color: #ffffff !important;
    }
    
    /* Customization for Streamlit Buttons */
    .stButton>button {
        background-color: #5c3593 !important;
        color: white !important;
        border: 1px solid #8a2be2 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #7944c3 !important;
        border-color: #bf80ff !important;
        box-shadow: 0px 0px 10px #8a2be2;
    }
    </style>
""", unsafe_allow_html=True)

# ================== SIDEBAR CONFIGURATION LOGIC ==================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌐 GlobalInternet.py</div>', unsafe_allow_html=True)
    
    # Language Selection Matrix Array
    language = st.selectbox(
        "🌐 Target Translation Array / Langue:",
        ["English", "Français", "Português"]
    )
    
    st.markdown("<hr style='border-color: #3a1f5d;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #bf80ff;'>🛠️ Core Infrastructure Desk</h3>", unsafe_allow_html=True)
    
    # User's Verified Corporate Information 
    st.markdown(f"""
    <div class="sidebar-info">
        <strong>Coder-in-Chief:</strong><br>
        <span style="color: #fff; font-weight:600;">Gesner Deslandes</span><br><br>
        <strong>Engineering Support Line:</strong><br>
        <span style="color: #bf80ff;">(509)-47385663</span><br><br>
        <strong>Status:</strong> <span style="color: #aa66ff;">● Production Live</span>
    </div>
    """, unsafe_allow_html=True)

# ================== LOCALIZATION DICTIONARY ==================
strings = {
    "English": {
        "title": "SignBridge AI Chatbot Platform",
        "sub": "Real-Time Sign Language Translation & Audio Sync Engine",
        "step1": "🎥 Step 1: Initialize Local Camera Stream Gateway",
        "step2": "💬 Step 2: Live Digital Translation Desk",
        "vector": "Translated Text Input Vector:",
        "btn": "Send to Chatbot Matrix",
        "bot_prefix": "System processed sign:",
        "bot_suffix": "How can I assist you further?",
        "voice": "en-US-ChristopherNeural"
    },
    "Français": {
        "title": "Plateforme Chatbot SignBridge AI",
        "sub": "Moteur de Traduction en Temps Réel et Synchronisation Audio",
        "step1": "🎥 Étape 1 : Initialiser la passerelle de flux caméra",
        "step2": "💬 Étape 2 : Bureau de traduction en direct",
        "vector": "Vecteur d'entrée de texte traduit :",
        "btn": "Envoyer au Chatbot Matrix",
        "bot_prefix": "Le système a traité le signe :",
        "bot_suffix": "Comment puis-je vous aider davantage ?",
        "voice": "fr-FR-HenriNeural"
    },
    "Português": {
        "title": "Plataforma Chatbot SignBridge AI",
        "sub": "Motor de Tradução em Tempo Real e Sincronização de Áudio",
        "step1": "🎥 Passo 1: Inicializar o Gateway de Transmissão da Câmera",
        "step2": "💬 Passo 2: Balcão de Tradução ao Vivo",
        "vector": "Vetor de Entrada de Texto Traduzido:",
        "btn": "Enviar para o Chatbot Matrix",
        "bot_prefix": "O sistema processou o sinal:",
        "bot_suffix": "Como posso ajudar você mais?",
        "voice": "pt-BR-AntonioNeural"
    }
}

current_lang = strings[language]

# Main Area Header Elements
st.markdown(f'<div class="main-header">{current_lang["title"]}</div>', unsafe_allow_html=True)
st.write(f"<i style='color:#aaa2bc;'>{current_lang['sub']}</i>", unsafe_allow_html=True)

# ================== MODERN MEDIAPIPE INITIALIZATION ==================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize persistent session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "detected_text" not in st.session_state:
    st.session_state.detected_text = ""

# --- AUDIO GENERATION FUNCTION ---
def generate_audio(text, voice_model):
    output_file = "response.mp3"
    communicate = edge_tts.Communicate(text, voice_model)
    asyncio.run(communicate.save(output_file))
    return output_file

# --- COMPUTER VISION VIDEO LAYER ---
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=None),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) 
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        detection_result = self.landmarker.detect(mp_image)
        
        detected_sign = ""
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Custom Purple Node tracing matrix loops (BGR format: Purple is Blue + Red)
                for landmark in hand_landmarks:
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    cv2.circle(img, (x, y), 5, (230, 30, 150), -1) # Glowing Purple/Magenta Nodes

                # Boundary Heuristics 
                thumb_tip = hand_landmarks[4].y
                index_tip = hand_landmarks[8].y
                if index_tip < thumb_tip:
                    detected_sign = "HELLO" if language == "English" else ("BONJOUR" if language == "Français" else "OLÁ")
                else:
                    detected_sign = "THANK YOU" if language == "English" else ("MERCI" if language == "Français" else "OBRIGADO")

            # Draw matching text banner onto the video stream
            cv2.putText(img, f"Sign: {detected_sign}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 120, 210), 2)
            
            st.session_state.detected_text = detected_sign

        return img

# --- CAMERA INPUT LAYER ---
st.write(f"<h3 style='color: #bf80ff;'>{current_lang['step1']}</h3>", unsafe_allow_html=True)
webrtc_streamer(
    key="sign-streamer-v3", 
    video_transformer_factory=SignLanguageTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

# --- CHATBOT INTERACTIVE INTERFACE ---
st.write(f"<h3 style='color: #bf80ff;'>{current_lang['step2']}</h3>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input(current_lang["vector"], value=st.session_state.detected_text)

with col2:
    st.write("##")
    send_btn = st.button(current_lang["btn"], use_container_width=True)

if send_btn and user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    
    bot_reply = f"{current_lang['bot_prefix']} '{user_input}'. {current_lang['bot_suffix']}"
    st.session_state.chat_history.append({"role": "bot", "text": bot_reply})
    
    audio_path = generate_audio(bot_reply, current_lang["voice"])
    st.session_state.latest_audio = audio_path

# --- DISPLAY STREAMLINED TRANSLATION TIMELINE ---
for msg in reversed(st.session_state.chat_history):
    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("assistant").write(msg["text"])
        if "latest_audio" in st.session_state and os.path.exists(st.session_state.latest_audio):
            st.audio(st.session_state.latest_audio, format="audio/mp3")

# ================== CORPORATE FOOTER DESK LAYER ==================
st.markdown("<hr style='border-color: #3a1f5d;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85rem; color:#aaa2bc; letter-spacing: 0.5px;">
        🚀 <strong>SIGNBRIDGE AI ENGINE</strong> | Developed and Maintained by <strong>GlobalInternet.py</strong><br>
        Architect-in-Chief: <strong>Gesner Deslandes</strong> | Infrastructure Desk Contact: <strong style="color:#bf80ff;">(509)-47385663</strong>
    </div>
    """,
    unsafe_allow_html=True
)
