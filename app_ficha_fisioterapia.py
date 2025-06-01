import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import av
import whisper
import numpy as np
from datetime import datetime
import os
import re
import pdfplumber

# --- Configurações iniciais da Página Streamlit ---
st.set_page_config(page_title="Ficha Atendimento - Fisioterapia", layout="centered")

# --- Definição dos Campos da Ficha e Ordem para Navegação ---
# Mapeia nomes amigáveis (que o profissional falará) para chaves de sessão (usadas no código)
# A ordem aqui define a sequência do comando "próximo campo"
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
    # Adicione mais campos se sua ficha tiver outros tópicos grandes de texto
}

# Lista ordenada das chaves dos campos para navegação "próximo campo"
FORM_FIELDS_ORDER = list(FORM_FIELDS_MAP.values())

# --- Inicialização de Estados da Sessão Streamlit ---
# Garante que as variáveis de estado existam para evitar KeyError

# Estado de login
if "logado" not in st.session_state:
    st.session_state.logado = False

# Armazenamento de fichas PDF padrão (nome amigável -> texto extraído)
if "fichas_pdf" not in st.session_state:
    st.session_state.fichas_pdf = {
        "ficha de anamnese": "Este é o modelo de texto da ficha de anamnese. Inclua histórico médico, queixas principais, cirurgias prévias e medicamentos em uso.",
        "ficha de avaliação ortopédica": "Este é o modelo de texto da ficha de avaliação ortopédica. Detalhe exame físico, testes específicos, postura, marcha e ADMs.",
        # Adicione outros modelos de ficha PDF aqui, eles serão carregados no início do app
    }

# Simulação de um "banco de dados" de pacientes e suas fichas
# Na vida real, isso viria de um banco de dados persistente ou API
if "pacientes" not in st.session_state:
    st.session_state.pacientes = {
        "joao silva": {"anamnese": "Histórico do João: dor no ombro direito há 3 meses, sem cirurgias, usa analgésicos esporadicamente.", "avaliacao_ortopedica": "João Silva, 45 anos, dor ao levantar o braço, ADM reduzida em abdução. Testes de manguito rotador positivos."},
        "maria souza": {"anamnese": "Histórico da Maria: lombalgia crônica, pós-operatório de hérnia de disco há 2 anos, faz uso contínuo de anti-inflamatórios.", "avaliacao_postural": "Maria Souza, 60 anos, assimetria pélvica, hipercifose torácica."},
    }

# Estado para controlar qual tipo de ficha (padrão/paciente/nova) está aberta
if "tipo_ficha_aberta" not in st.session_state:
    st.session_state.tipo_ficha_aberta = None

# Estado para armazenar o paciente atualmente selecionado
if "paciente_atual" not in st.session_state:
    st.session_state.paciente_atual = None

# Estado para o ditado geral (se não houver campo específico ativo)
if "transcricao_geral" not in st.session_state:
    st.session_state.transcricao_geral = ""

# Estado para exibir o último segmento transcrito (feedback visual)
if "last_transcription_segment" not in st.session_state:
    st.session_state.last_transcription_segment = ""

# Estado para controlar qual campo do formulário está ativo para ditado por voz
if "active_form_field" not in st.session_state:
    st.session_state.active_form_field = None # Nenhum campo ativo inicialmente

# NOVO: Estado para controlar se a anotação por voz está ativa ou pausada
if "listening_active" not in st.session_state:
    st.session_state.listening_active = True # Começa escutando

# NOVO: Estado para a mensagem de status do microfone
if "mic_status_message" not in st.session_state:
    st.session_state.mic_status_message = "🔴 Microfone Desconectado"

# NOVO: Estado para controlar a inicialização do webrtc_streamer
if "webrtc_initialized" not in st.session_state:
    st.session_state.webrtc_initialized = False


# Inicializa todos os campos do formulário definidos em FORM_FIELDS_MAP
for key in FORM_FIELDS_MAP.values():
    if key not in st.session_state:
        st.session_state[key] = "" # Cada campo começa como string vazia

# --- Funções de Login ---
def login():
    st.title("🔐 Login")
    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fisioterapeuta" and pwd == "1234":
            st.session_state.logado = True
            st.rerun() # Atualiza a página após o login
        else:
            st.error("Usuário ou senha incorretos")

# Se não estiver logado, exibe a tela de login e para a execução do restante do app
if not st.session_state.logado:
    login()
    st.stop()

# --- Carregar modelo Whisper uma única vez (em cache para performance) ---
@st.cache_resource
def carregar_modelo():
    st.info("Carregando modelo Whisper (pode levar alguns segundos)...")
    # Tenta carregar o modelo "base.en" primeiro, depois "base"
    try:
        model = whisper.load_model("base.en") # Pode ser mais leve e suficiente para comandos em inglês
        st.success("Modelo Whisper 'base.en' carregado!")
    except Exception as e:
        st.warning(f"Não foi possível carregar 'base.en': {e}. Tentando 'base' (maior).")
        model = whisper.load_model("base") # Carrega o modelo "base" do Whisper
        st.success("Modelo Whisper 'base' carregado!")
    return model

model = carregar_modelo()

# --- Função para correção de termos comuns na transcrição ---
def corrigir_termos(texto):
    correcoes = {
        "tendinite": "tendinite",
        "cervicalgia": "cervicalgia",
        "lombar": "região lombar",
        "reabilitação funcional": "reabilitação funcional",
        "fisioterapia do ombro": "fisioterapia de ombro",
        "dor nas costas": "algia na coluna",
        # Adicione mais correções conforme necessário
    }
    # Aplica as correções. O replace é case-sensitive, então a transcrição é normalizada para minúsculas antes.
    for errado, certo in correcoes.items():
        texto = texto.replace(errado, certo)
    return texto

# --- CLASSE PARA PROCESSAMENTO DE ÁUDIO (CORE DA TRANSCRIÇÃO E COMANDOS DE VOZ) ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffer = b"" # Buffer para acumular áudio

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.float32).tobytes()
        self.buffer += pcm

        # Processa o buffer a cada 5 segundos de áudio acumulado
        if len(self.buffer) > 32000 * 5: # 32000 samples/seg * 5 segundos
            audio_np = np.frombuffer(self.buffer, np.float32)
            audio_np = whisper.pad_or_trim(audio_np) # Ajusta o tamanho do áudio para o Whisper
            mel = whisper.log_mel_spectrogram(audio_np).to(model.device) # Converte para mel spectrogram
            options = whisper.DecodingOptions(language="pt", fp16=False) # Opções de decodificação (português)
            result = whisper.decode(model, mel, options) # Decodifica o áudio para texto
            
            # Corrige e normaliza o texto transcrito
            texto_transcrito_segmento = corrigir_terms(result.text).strip()
            st.session_state.last_transcription_segment = texto_transcrito_segmento # Atualiza feedback visual
            
            comando_processado = False
            texto_transcrito_lower = texto_transcrito_segmento.lower() # Para facilitar a comparação de comandos

            # --- NOVO: Lógica de Comandos de Pausa/Retomada ---
            if "pausar anotação" in texto_transcrito_lower:
                st.session_state.listening_active = False
                st.session_state.last_transcription_segment = "" # Limpa para não mostrar o comando
                # Não st.rerun() aqui, o feedback será dado pela UI
                comando_processado = True
            elif "retomar anotação" in texto_transcrito_lower:
                st.session_state.listening_active = True
                st.session_state.last_transcription_segment = "" # Limpa para não mostrar o comando
                # Não st.rerun() aqui, o feedback será dado pela UI
                comando_processado = True

            # --- Lógica de Comandos de Voz de Abertura de Ficha ---
            # Estes comandos ainda forçarão um rerun, causando uma pequena interrupção no áudio.
            # É uma limitação atual para garantir que a UI se atualize com a nova ficha.

            # Comando: "abrir ficha de [tipo da ficha padrão]" ou "mostrar [tipo da ficha padrão]"
            match_abrir_ficha_padrao = re.search(r"(?:abrir|mostrar) ficha de (.+)", texto_transcrito_lower)
            if match_abrir_ficha_padrao and not comando_processado:
                ficha_solicitada = match_abrir_ficha_padrao.group(1).strip()
                if ficha_solicitada in st.session_state.fichas_pdf:
                    st.session_state.paciente_atual = None # Reseta o paciente atual ao abrir ficha padrão
                    st.session_state.tipo_ficha_aberta = ficha_solicitada # Define o tipo de ficha aberta
                    st.session_state.transcricao_geral = st.session_state.fichas_pdf[ficha_solicitada] # Carrega o texto da ficha no campo geral
                    
                    # NOVO: Ao abrir uma ficha padrão, resetar os campos específicos do formulário
                    for key in FORM_FIELDS_MAP.values():
                        st.session_state[key] = "" 
                    
                    st.session_state.active_form_field = None # Reseta campo de formulário ativo
                    st.success(f"Ficha padrão '{ficha_solicitada.title()}' aberta e texto carregado!")
                    st.rerun() # Força a atualização da UI
                    comando_processado = True
                else:
                    st.warning(f"Comando de ficha padrão '{ficha_solicitada}' não reconhecido.")
                    # Não set comando_processado=True aqui se não achou a ficha,
                    # para permitir que a transcrição normal continue, caso seja apenas uma frase.

            # Comando: "abrir ficha do paciente [nome do paciente] (de|da)? [tipo da ficha]"
            match_abrir_paciente_ficha = re.search(r"abrir ficha do paciente (.+?) (?:de|da)? (.+)", texto_transcrito_lower)
            if match_abrir_paciente_ficha and not comando_processado:
                nome_paciente_falado = match_abrir_paciente_ficha.group(1).strip()
                tipo_ficha_falado = match_abrir_paciente_ficha.group(2).strip()
                
                found_patient = None
                # Busca flexível: verifica se o nome falado está contido no nome do paciente no "DB"
                for p_name_db in st.session_state.pacientes:
                    if nome_paciente_falado in p_name_db: 
                        found_patient = p_name_db
                        break

                if found_patient:
                    if tipo_ficha_falado in st.session_state.pacientes[found_patient]:
                        st.session_state.paciente_atual = found_patient
                        st.session_state.tipo_ficha_aberta = tipo_ficha_falado
                        # NOVO: Carrega o texto da ficha do paciente diretamente para o campo geral para o prazo de hoje.
                        # O mapeamento para campos específicos é mais complexo.
                        st.session_state.transcricao_geral = st.session_state.pacientes[found_patient][tipo_ficha_falado]
                        
                        # NOVO: Limpa os campos específicos do formulário ao abrir ficha de paciente
                        for key in FORM_FIELDS_MAP.values():
                            st.session_state[key] = ""
                        
                        st.session_state.active_form_field = None # Reseta campo de formulário ativo
                        st.success(f"Ficha '{tipo_ficha_falado.title()}' do paciente '{found_patient.title()}' aberta e texto carregado!")
                        st.rerun()
                        comando_processado = True
                    else:
                        st.warning(f"Não foi possível encontrar a ficha '{tipo_ficha_falado.title()}' para o paciente '{found_patient.title()}'.")
                        comando_processado = True # Processou o comando, mas com erro
                else:
                    st.warning(f"Paciente '{nome_paciente_falado.title()}' não encontrado.")
                    comando_processado = True # Processou o comando, mas com erro

            # Comando: "nova ficha de [tipo da ficha]"
            match_nova_ficha = re.search(r"nova ficha de (.+)", texto_transcrito_lower)
            if match_nova_ficha and not comando_processado:
                tipo_nova_ficha = match_nova_ficha.group(1).strip()
                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = f"Nova: {tipo_nova_ficha}"
                st.session_state.transcricao_geral = "" # Zera a transcrição geral para nova ficha
                
                # NOVO: Zera todos os campos do formulário para nova ficha
                for key in FORM_FIELDS_MAP.values():
                    st.session_state[key] = ""
                
                st.session_state.active_form_field = FORM_FIELDS_ORDER[0] if FORM_FIELDS_ORDER else None # Ativa o primeiro campo
                if st.session_state.active_form_field:
                    st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Comece a ditar no campo **{st.session_state.active_form_field.replace('_', ' ').title()}**.")
                else:
                    st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Não há campos definidos. Dite em observações gerais.")
                st.rerun()
                comando_processado = True

            # --- Lógica de Comandos de Voz para Navegação entre Campos do Formulário ---

            # Comando: "Preencher [nome do campo]"
            match_preencher_campo = re.search(r"preencher (.+)", texto_transcrito_lower)
            if match_preencher_campo and not comando_processado:
                campo_falado = match_preencher_campo.group(1).strip()
                found_field_key = None
                # Busca o campo que mais se assemelha ou que contém a palavra chave
                for friendly_name, field_key in FORM_FIELDS_MAP.items():
                    if campo_falado in friendly_name: # Busca parcial para maior flexibilidade
                        found_field_key = field_key
                        break
                
                if found_field_key:
                    st.session_state.active_form_field = found_field_key
                    # st.info já é atualizado na UI, não precisamos de rerun aqui se o estado já foi setado
                    comando_processado = True
                    st.session_state.last_transcription_segment = "" # Limpa o segmento para não adicionar o comando ao campo
                    st.rerun() # Ainda necessário para forçar o destaque
