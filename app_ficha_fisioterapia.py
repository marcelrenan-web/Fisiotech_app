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

# --- Funções Auxiliares (mantidas ou levemente ajustadas) ---

@st.cache_data(show_spinner="Extraindo texto do PDF...")
def read_pdf_text(file_path):
    if not os.path.exists(file_path):
        #st.error(f"Erro: Arquivo PDF não encontrado em '{file_path}'") # Comentar para evitar erro em ficheiro opcional
        return ""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
