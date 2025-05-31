import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode # <<< ADICIONE ISSO
import av
import whisper
import numpy as np
from datetime import datetime
import os

# --- Configurações iniciais ---
st.set_page_config(page_title="Ficha Atendimento - Fisioterapia", layout="centered")

# --- Login simples ---
if "logado" not in st.session_state:
    st.session_state.logado = False

def login():
    st.title("🔐 Login")
    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fisioterapeuta" and pwd == "1234":
            st.session_state.logado = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos")

if not st.session_state.logado:
    login()
    st.stop()

# --- Carregar modelo Whisper uma única vez ---
@st.cache_resource
def carregar_modelo():
    return whisper.load_model("base")

model = carregar_modelo()

# --- Correção de termos comuns ---
def corrigir_termos(texto):
    correcoes = {
        "tendinite": "tendinite",
        "cervicalgia": "cervicalgia",
        "lombar": "região lombar",
        "reabilitação funcional": "reabilitação funcional",
        "fisioterapia do ombro": "fisioterapia de ombro",
        "dor nas costas": "algia na coluna",
    }
    for errado, certo in correcoes.items():
        texto = texto.replace(errado, certo)
    return texto

# --- Inicializar transcrição ---
if "transcricao" not in st.session_state:
    st.session_state.transcricao = ""

# --- Processador de áudio ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffer = b""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.float32).tobytes()
        self.buffer += pcm

        if len(self.buffer) > 32000 * 5:  # 5 segundos
            audio_np = np.frombuffer(self.buffer, np.float32)
            audio_np = whisper.pad_or_trim(audio_np)
            mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
            options = whisper.DecodingOptions(language="pt", fp16=False)
            result = whisper.decode(model, mel, options)
            texto_corrigido = corrigir_termos(result.text)
            st.session_state.transcricao += texto_corrigido + " "
            self.buffer = b""
        return frame

# --- Interface ---
st.title("🩺 Ficha de Atendimento - Fisioterapia com IA")

st.subheader("🎤 Fale e veja o texto ao vivo:")
webrtc_streamer(
    key="microfone",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

st.text_area("📝 Texto reconhecido:", st.session_state.transcricao, height=200)

st.subheader("📋 Preencha os dados do atendimento")

with st.form("form_ficha"):
    nome = st.text_input("Nome do paciente")
    idade = st.number_input("Idade", min_value=0, max_value=120)
    data = st.date_input("Data do atendimento", value=datetime.today())
    sintomas = st.text_area("Relato do paciente", value=st.session_state.transcricao)
    diagnostico = st.text_area("Diagnóstico clínico")
    conduta = st.text_area("Conduta adotada")
    enviar = st.form_submit_button("Salvar ficha")

    if enviar:
        # Criar pasta se não existir
        pasta = "fichas_salvas"
        if not os.path.exists(pasta):
            os.makedirs(pasta)
        # Criar nome único para o arquivo
        nome_arquivo = f"{pasta}/ficha_{nome.replace(' ', '_')}_{data.strftime('%Y%m%d')}.txt"
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            f.write(f"Paciente: {nome}\n")
            f.write(f"Idade: {idade} anos\n")
            f.write(f"Data: {data.strftime('%d/%m/%Y')}\n")
            f.write(f"Relato: {sintomas}\n")
            f.write(f"Diagnóstico: {diagnostico}\n")
            f.write(f"Conduta: {conduta}\n")
        st.success(f"✅ Ficha salva em '{nome_arquivo}'")
        # Limpar transcrição e campos
        st.session_state.transcricao = ""
