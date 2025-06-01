import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import torch
import whisper
from googletrans import Translator
import time

# Inicialização de sessão
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "listening_active" not in st.session_state:
    st.session_state.listening_active = False
if "mic_status_message" not in st.session_state:
    st.session_state.mic_status_message = "🔴 Microfone Desativado"

# Inicialização do modelo Whisper
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()
translator = Translator()

# Função para corrigir termos comuns em fisioterapia
def corrigir_termos(texto):
    substituicoes = {
        "palmo": "palpação",
        "pamo": "palpação",
        "algia": "dor",
        "alvo": "álvo",
        "rom": "amplitude de movimento",
        "rôm": "amplitude de movimento",
        "robin": "rubbing",
        "propriocepção": "propriocepção",
        "estiramento": "alongamento"
    }
    for termo_errado, termo_correto in substituicoes.items():
        texto = texto.replace(termo_errado, termo_correto)
    return texto

# Processador de áudio
class AudioProcessor:
    def __init__(self):
        self.buffer = b""
        self.last_transcription_time = time.time()

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten()
        audio_tensor = torch.from_numpy(audio).float() / 32768.0

        # Verifica tempo mínimo entre transcrições (0.5s)
        if st.session_state.listening_active and time.time() - self.last_transcription_time > 0.5:
            try:
                result = model.transcribe(audio_tensor.numpy(), language="pt")
                texto_transcrito_segmento = corrigir_termos(result["text"]).strip()

                if texto_transcrito_segmento:
                    st.session_state.transcription += " " + texto_transcrito_segmento
                    self.last_transcription_time = time.time()
            except Exception as e:
                st.warning(f"Erro na transcrição: {e}")

        return frame

# Interface do Streamlit
st.title("🩺 Assistente de Transcrição para Fisioterapia")
st.markdown("Transcreva automaticamente os atendimentos por voz utilizando inteligência artificial.")

# Botão para iniciar/parar escuta
if st.button("🎙️ Iniciar/Pausar Microfone"):
    st.session_state.listening_active = not st.session_state.listening_active
    if st.session_state.listening_active:
        st.session_state.mic_status_message = "🟢 Microfone Ativo"
    else:
        st.session_state.mic_status_message = "🟠 Anotação Pausada"

# Status do microfone
st.markdown(f"### {st.session_state.mic_status_message}")

# Inicialização do WebRTC
webrtc_ctx = webrtc_streamer(
    key="fisioterapia-audio",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_processor_factory=AudioProcessor,
)

# Exibição da transcrição
st.subheader("📝 Transcrição")
st.text_area("Texto transcrito:", st.session_state.transcription, height=300)

# Botão para limpar
if st.button("🧹 Limpar Transcrição"):
    st.session_state.transcription = ""
