import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import av
import whisper
import numpy as np
from datetime import datetime
import os
import re
import pdfplumber
import fitz # PyMuPDF
from PIL import Image
import io # Para lidar com bytes de imagem
import json # Para salvar metadados de fichas uploadadas

# --- Configurações iniciais da Página Streamlit ---
st.set_page_config(page_title="Ficha Atendimento - Fisioterapia", layout="centered")

# --- Definição dos Campos da Ficha e Ordem para Navegação ---
FORM_FIELDS_MAP = {
    "localizacao e caracteristicas do sintoma": "sintomas_localizacao",
    "cirurgias traumatismos parto": "cirurgias_traumatismos_parto",
    "sono": "sono",
    "atividade fisica": "atividade_fisica",
    "condicoes gerais": "condicoes_gerais",
    "alimentacao": "alimentacao",
    "alergias": "alergias",
    "emocional": "emocional",
    "medicacao": "medicacao",
}

FORM_FIELDS_ORDER = list(FORM_FIELDS_MAP.values())

# --- Caminhos para Armazenamento ---
# Ajustado para usar 'dados/' conforme sua estrutura de pastas no GitHub
UPLOADED_TEMPLATES_DIR = "dados/uploaded_fichas_templates"
UPLOADED_TEMPLATES_INDEX_FILE = "dados/uploaded_fichas_index.json"
SAVED_RECORDS_DIR = "dados/saved_patient_records" # Para fichas preenchidas

# Garante que os diretórios existam
os.makedirs(UPLOADED_TEMPLATES_DIR, exist_ok=True)
os.makedirs(SAVED_RECORDS_DIR, exist_ok=True)

# --- Funções para Persistência de Fichas Uploadadas ---
def load_uploaded_templates_index():
    if os.path.exists(UPLOADED_TEMPLATES_INDEX_FILE):
        try:
            with open(UPLOADED_TEMPLATES_INDEX_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Se o JSON estiver corrompido ou vazio, retorna um dicionário vazio
            st.warning(f"Erro ao ler {UPLOADED_TEMPLATES_INDEX_FILE}. Criando um novo.")
            return {}
    return {}

def save_uploaded_templates_index(index_data):
    with open(UPLOADED_TEMPLATES_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4, ensure_ascii=False)

# --- Inicialização de Estados da Sessão Streamlit ---
if "logado" not in st.session_state:
    st.session_state.logado = False

# Fichas padrão (as que vêm com o app, ex: a de avaliação que você enviou)
if "fichas_padrao_paths" not in st.session_state:
    st.session_state.fichas_padrao_paths = {
        "ficha de anamnese": "ficha_anamnese_padrao_exemplo.pdf", # Placeholder: se você tiver este PDF, coloque na raiz
        "ficha de avaliação ortopédica": "Ficha de Avaliação de Fisioterapia nova - Documentos Google.pdf", # Deve estar na raiz
    }

# NOVO: Carrega as fichas uploadadas e salvas
if "uploaded_fichas_data" not in st.session_state:
    st.session_state.uploaded_fichas_data = load_uploaded_templates_index()

# Cache para o texto e imagens do PDF
if "fichas_pdf_content_cache" not in st.session_state:
    st.session_state.fichas_pdf_content_cache = {}
if "fichas_pdf_images_cache" not in st.session_state:
    st.session_state.fichas_pdf_images_cache = {}

if "pacientes" not in st.session_state:
    st.session_state.pacientes = {
        "joao silva": {"anamnese": "Histórico do João: dor no ombro direito há 3 meses, sem cirurgias, usa analgésicos esporadicamente.", "avaliacao_ortopedica": "João Silva, 45 anos, dor ao levantar o braço, ADM reduzida em abdução. Testes de manguito rotador positivos."},
        "maria souza": {"anamnese": "Histórico da Maria: lombalgia crônica, pós-operatório de hérnia de disco há 2 anos, faz uso contínuo de anti-inflamatórios.", "avaliacao_postural": "Maria Souza, 60 anos, assimetria pélvica, hipercifose torácica."},
    }

if "tipo_ficha_aberta" not in st.session_state:
    st.session_state.tipo_ficha_aberta = None

if "paciente_atual" not in st.session_state:
    st.session_state.paciente_atual = None

if "transcricao_geral" not in st.session_state:
    st.session_state.transcricao_geral = ""

if "current_pdf_images" not in st.session_state:
    st.session_state.current_pdf_images = []

if "last_transcription_segment" not in st.session_state:
    st.session_state.last_transcription_segment = ""

if "active_form_field" not in st.session_state:
    st.session_state.active_form_field = None

if "listening_active" not in st.session_state:
    st.session_state.listening_active = True

if "mic_status_message" not in st.session_state:
    st.session_state.mic_status_message = "🔴 Microfone Desconectado"

if "webrtc_initialized" not in st.session_state:
    st.session_state.webrtc_initialized = False

for key in FORM_FIELDS_MAP.values():
    if key not in st.session_state:
        st.session_state[key] = ""

# --- Funções Auxiliares ---

@st.cache_data(show_spinner="Extraindo texto do PDF...")
def read_pdf_text(file_path):
    if not os.path.exists(file_path):
        #st.error(f"Erro: Arquivo PDF não encontrado em '{file_path}'") # Comentar para evitar erro em ficheiro opcional
        return ""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text(x_tolerance=2) + "\n"
        return text
    except Exception as e:
        st.error(f"Erro ao ler PDF '{file_path}': {e}")
        return ""

@st.cache_data(show_spinner="Preparando visualização do PDF...")
def get_pdf_images(file_path):
    if not os.path.exists(file_path):
        #st.error(f"Erro: Arquivo PDF não encontrado em '{file_path}'") # Comentar para evitar erro em ficheiro opcional
        return []
    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.pil_tobytes(format="PNG")
            images.append(Image.open(io.BytesIO(img_bytes)))
        doc.close()
        return images
    except Exception as e:
        st.error(f"Erro ao converter PDF para imagem: {e}")
        return []

def login_page():
    st.title("🔐 Login")
    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fisioterapeuta" and pwd == "1234":
            st.session_state.logado = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos")

# --- Lógica Principal do Aplicativo ---
if not st.session_state.logado:
    login_page()
else:
    @st.cache_resource
    def carregar_modelo():
        st.info("Carregando modelo Whisper (pode levar alguns segundos)...")
        try:
            model = whisper.load_model("base.en")
            st.success("Modelo Whisper 'base.en' carregado!")
        except Exception as e:
            st.warning(f"Não foi possível carregar 'base.en': {e}. Tentando 'base' (maior).")
            model = whisper.load_model("base")
            st.success("Modelo Whisper 'base' carregado!")
        return model

    model = carregar_modelo()

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

    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.buffer = b""

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray().flatten().astype(np.float32).tobytes()
            self.buffer += pcm

            if len(self.buffer) > 32000 * 5:
                audio_np = np.frombuffer(self.buffer, np.float32)
                audio_np = whisper.pad_or_trim(audio_np)
                mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
                options = whisper.DecodingOptions(language="pt", fp16=False)
                result = whisper.decode(model, mel, options)
                
                texto_transcrito_segmento = corrigir_termos(result.text).strip()
                st.session_state.last_transcription_segment = texto_transcrito_segmento
                
                comando_processado = False
                texto_transcrito_lower = texto_transcrito_segmento.lower()

                if "pausar anotação" in texto_transcrito_lower:
                    st.session_state.listening_active = False
                    st.session_state.last_transcription_segment = ""
                    comando_processado = True
                elif "retomar anotação" in texto_transcrito_lower:
                    st.session_state.listening_active = True
                    st.session_state.last_transcription_segment = ""
                    comando_processado = True

                # Lógica para abrir PDF via comando de voz (PADRÃO E UPLOADED)
                match_abrir_ficha_padrao_ou_upload = re.search(r"(?:abrir|mostrar) ficha de (.+)", texto_transcrito_lower)
                if match_abrir_ficha_padrao_ou_upload and not comando_processado:
                    ficha_solicitada = match_abrir_ficha_padrao_ou_upload.group(1).strip()
                    file_path_to_open = None

                    # Tenta encontrar em fichas padrão
                    if ficha_solicitada in st.session_state.fichas_padrao_paths:
                        file_path_to_open = st.session_state.fichas_padrao_paths[ficha_solicitada]
                    # Tenta encontrar em fichas uploadadas
                    elif ficha_solicitada in st.session_state.uploaded_fichas_data:
                        file_path_to_open = st.session_state.uploaded_fichas_data[ficha_solicitada]["path"]
                    
                    if file_path_to_open:
                        # Carregamento de texto (não usado para o campo editável, mas pode ser útil para outras coisas)
                        if file_path_to_open not in st.session_state.fichas_pdf_content_cache:
                            st.session_state.fichas_pdf_content_cache[file_path_to_open] = read_pdf_text(file_path_to_open)
                        
                        # Carregamento das imagens do PDF
                        if file_path_to_open not in st.session_state.fichas_pdf_images_cache:
                            st.session_state.fichas_pdf_images_cache[file_path_to_open] = get_pdf_images(file_path_to_open)
                        st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[file_path_to_open]

                        st.session_state.paciente_atual = None
                        st.session_state.tipo_ficha_aberta = ficha_solicitada
                        st.session_state.transcricao_geral = "" # Zera para as respostas
                        
                        for key in FORM_FIELDS_MAP.values():
                            st.session_state[key] = "" 
                        
                        st.session_state.active_form_field = None
                        st.success(f"Ficha '{ficha_solicitada.title()}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                        st.rerun()
                        comando_processado = True
                    else:
                        st.warning(f"Comando de ficha '{ficha_solicitada}' não reconhecido.")

                # O restante da lógica de comandos de voz (abrir paciente, nova ficha, preencher, proximo, anterior) permanece a mesma
                match_abrir_paciente_ficha = re.search(r"abrir ficha do paciente (.+?) (?:de|da)? (.+)", texto_transcrito_lower)
                if match_abrir_paciente_ficha and not comando_processado:
                    nome_paciente_falado = match_abrir_paciente_ficha.group(1).strip()
                    tipo_ficha_falado = match_abrir_paciente_ficha.group(2).strip()
                    
                    found_patient = None
                    for p_name_db in st.session_state.pacientes:
                        if nome_paciente_falado in p_name_db:
