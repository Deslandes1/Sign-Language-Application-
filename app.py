import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import edge_tts
import os

# --- INITIALIZATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

st.title("🌐 SignBridge AI Chatbot — GlobalInternet.py")
st.subheader("Real-Time Sign Language Translation & Audio Sync")

# Initialize persistent session states for the chatbot conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "detected_text" not in st.session_state:
    st.session_state.detected_text = ""

# --- AUDIO GENERATION FUNCTION ---
def generate_audio(text, lang="en-US"):
    """Generates fluid TTS output using edge-tts."""
    voice = "en-US-ChristopherNeural" if lang == "en-US" else "fr-FR-HenriNeural"
    output_file = "response.mp3"
    
    communicate = edge_tts.Communicate(text, voice)
    asyncio.run(communicate.save(output_file))
    return output_file

# --- COMPUTER VISION VIDEO LAYER ---
class SignLanguageTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Flip image horizontally for natural mirroring effect
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process landmarks
        results = hands.process(rgb_img)
        
        detected_sign = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton map on screen
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # --- SIGN TRANSLATION LOGIC ---
                # Extract landmark coordinates for model classification
                landmarks = [lm.x for lm in hand_landmarks.landmark]
                
                # Example Placeholder Logic: Simple boundary heuristic for testing
                # Real deployment will swap this with your trained model array (.pkl / .h5)
                thumb_tip = hand_landmarks.landmark[4].y
                index_tip = hand_landmarks.landmark[8].y
                if index_tip < thumb_tip:
                    detected_sign = "HELLO"
                else:
                    detected_sign = "THANK YOU"

            # Overlay detected text on the camera feed frame
            cv2.putText(img, f"Sign: {detected_sign}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update session state with the current stable sign
            st.session_state.detected_text = detected_sign

        return img

# --- CAMERA INPUT FIELD ---
st.write("### 🎥 Step 1: Initialize Camera Stream")
webrtc_streamer(
    key="sign-streamer", 
    video_transformer_factory=SignLanguageTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False} # No mic needed, output only
)

# --- CHATBOT INTERACTIVE INTERFACE ---
st.write("### 💬 Step 2: Live Translation Desk")

col1, col2 = st.columns([3, 1])
with col1:
    # Captures current word or lets user override via typing
    user_input = st.text_input("Translated Text Input Vector:", value=st.session_state.detected_text)

with col2:
    st.write("##")
    send_btn = st.button("Send to Chatbot", use_container_width=True)

if send_btn and user_input:
    # Append user's translated sign to log
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    
    # Generate bot reply (Integrating a quick mock response or LLM hook)
    bot_reply = f"System processed sign: '{user_input}'. How can I assist you further?"
    st.session_state.chat_history.append({"role": "bot", "text": bot_reply})
    
    # Render and store raw audio file
    audio_path = generate_audio(bot_reply)
    st.session_state.latest_audio = audio_path

# --- DISPLAY STREAMLINED CONVERSATION ---
for msg in reversed(st.session_state.chat_history):
    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("assistant").write(msg["text"])
        if "latest_audio" in st.session_state and os.path.exists(st.session_state.latest_audio):
            st.audio(st.session_state.latest_audio, format="audio/mp3")
