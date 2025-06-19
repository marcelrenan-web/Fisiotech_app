import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import av
import whisper
import numpy as np
from datetime import datetime
import os
import re
import pdfplumber
import fitz # Importa PyMuPDF para lidar com imagens de PDF
from PIL import Image # Para manipular imagens, caso necessário

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

# --- Inicialização de Estados da Sessão Streamlit ---
if "logado" not in st.session_state:
    st.session_state.logado = False

# Armazenamento de caminhos de arquivos PDF padrão
if "fichas_pdf_paths" not in st.session_state:
    st.session_state.fichas_pdf_paths = {
        "ficha de anamnese": "ficha_anamnese_padrao_exemplo.pdf", # Substitua pelo seu PDF real
        "ficha de avaliação ortopédica": "Ficha de Avaliação de Fisioterapia nova - Documentos Google.pdf",
    }
# Cache para o texto do PDF
if "fichas_pdf_content_cache" not in st.session_state:
    st.session_state.fichas_pdf_content_cache = {}
# Cache para as imagens do PDF
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

# NOVO: Estado para armazenar as imagens do PDF aberto (para visualização)
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
        st.error(f"Erro: Arquivo PDF não encontrado em '{file_path}'")
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

# NOVA FUNÇÃO: Para extrair imagens de cada página do PDF
@st.cache_data(show_spinner="Preparando visualização do PDF...")
def get_pdf_images(file_path):
    if not os.path.exists(file_path):
        st.error(f"Erro: Arquivo PDF não encontrado em '{file_path}'")
        return []
    
    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.pil_tobytes(format="PNG") # Converte para bytes de imagem PNG
            images.append(Image.open(io.BytesIO(img_bytes))) # Abre com PIL para Streamlit
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

                match_abrir_ficha_padrao = re.search(r"(?:abrir|mostrar) ficha de (.+)", texto_transcrito_lower)
                if match_abrir_ficha_padrao and not comando_processado:
                    ficha_solicitada = match_abrir_ficha_padrao.group(1).strip()
                    if ficha_solicitada in st.session_state.fichas_pdf_paths:
                        file_path = st.session_state.fichas_pdf_paths[ficha_solicitada]
                        
                        # Cache e carregamento do texto
                        if file_path not in st.session_state.fichas_pdf_content_cache:
                            st.session_state.fichas_pdf_content_cache[file_path] = read_pdf_text(file_path)
                        # st.session_state.transcricao_geral = st.session_state.fichas_pdf_content_cache[file_path] # AGORA COMEÇA VAZIO

                        # Cache e carregamento das imagens do PDF
                        if file_path not in st.session_state.fichas_pdf_images_cache:
                            st.session_state.fichas_pdf_images_cache[file_path] = get_pdf_images(file_path)
                        st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[file_path]

                        st.session_state.paciente_atual = None
                        st.session_state.tipo_ficha_aberta = ficha_solicitada
                        st.session_state.transcricao_geral = "" # AQUI: Zera para as respostas, não o texto do PDF
                        
                        for key in FORM_FIELDS_MAP.values():
                            st.session_state[key] = "" 
                        
                        st.session_state.active_form_field = None
                        st.success(f"Ficha padrão '{ficha_solicitada.title()}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                        st.rerun()
                        comando_processado = True
                    else:
                        st.warning(f"Comando de ficha padrão '{ficha_solicitada}' não reconhecido.")

                # O restante da lógica de comandos de voz (abrir paciente, nova ficha, preencher, proximo, anterior) permanece a mesma
                match_abrir_paciente_ficha = re.search(r"abrir ficha do paciente (.+?) (?:de|da)? (.+)", texto_transcrito_lower)
                if match_abrir_paciente_ficha and not comando_processado:
                    nome_paciente_falado = match_abrir_paciente_ficha.group(1).strip()
                    tipo_ficha_falado = match_abrir_paciente_ficha.group(2).strip()
                    
                    found_patient = None
                    for p_name_db in st.session_state.pacientes:
                        if nome_paciente_falado in p_name_db: 
                            found_patient = p_name_db
                            break

                    if found_patient:
                        if tipo_ficha_falado in st.session_state.pacientes[found_patient]:
                            st.session_state.paciente_atual = found_patient
                            st.session_state.tipo_ficha_aberta = tipo_ficha_falado
                            st.session_state.transcricao_geral = st.session_state.pacientes[found_patient][tipo_ficha_falado]
                            
                            st.session_state.current_pdf_images = [] # Limpa imagens de PDF ao abrir ficha de paciente
                            
                            for key in FORM_FIELDS_MAP.values():
                                st.session_state[key] = ""
                            
                            st.session_state.active_form_field = None
                            st.success(f"Ficha '{tipo_ficha_falado.title()}' do paciente '{found_patient.title()}' aberta e texto carregado!")
                            st.rerun()
                            comando_processado = True
                        else:
                            st.warning(f"Não foi possível encontrar a ficha '{tipo_ficha_falado.title()}' para o paciente '{found_patient.title()}'.")
                            comando_processado = True
                    else:
                        st.warning(f"Paciente '{nome_paciente_falado.title()}' não encontrado.")
                        comando_processado = True

                match_nova_ficha = re.search(r"nova ficha de (.+)", texto_transcrito_lower)
                if match_nova_ficha and not comando_processado:
                    tipo_nova_ficha = match_nova_ficha.group(1).strip()
                    st.session_state.paciente_atual = None
                    st.session_state.tipo_ficha_aberta = f"Nova: {tipo_nova_ficha}"
                    st.session_state.transcricao_geral = ""
                    
                    st.session_state.current_pdf_images = [] # Limpa imagens de PDF para nova ficha
                    
                    for key in FORM_FIELDS_MAP.values():
                        st.session_state[key] = ""
                    
                    st.session_state.active_form_field = FORM_FIELDS_ORDER[0] if FORM_FIELDS_ORDER else None
                    if st.session_state.active_form_field:
                        st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Comece a ditar no campo **{st.session_state.active_form_field.replace('_', ' ').title()}**.")
                    else:
                        st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Não há campos definidos. Dite em observações gerais.")
                    st.rerun()
                    comando_processado = True

                match_preencher_campo = re.search(r"preencher (.+)", texto_transcrito_lower)
                if match_preencher_campo and not comando_processado:
                    campo_falado = match_preencher_campo.group(1).strip()
                    found_field_key = None
                    for friendly_name, field_key in FORM_FIELDS_MAP.items():
                        if campo_falado in friendly_name:
                            found_field_key = field_key
                            break
                    
                    if found_field_key:
                        st.session_state.active_form_field = found_field_key
                        comando_processado = True
                        st.session_state.last_transcription_segment = ""
                        st.rerun()
                    else:
                        st.warning(f"Campo '{campo_falado.title()}' não reconhecido.")
                        comando_processado = True

                if "proximo campo" in texto_transcrito_lower and not comando_processado:
                    if st.session_state.active_form_field:
                        current_index = FORM_FIELDS_ORDER.index(st.session_state.active_form_field)
                        if current_index < len(FORM_FIELDS_ORDER) - 1:
                            st.session_state.active_form_field = FORM_FIELDS_ORDER[current_index + 1]
                            st.info(f"Campo ativo alterado para: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                        else:
                            st.info("Você está no último campo do formulário.")
                            st.session_state.active_form_field = None
                    else:
                        if FORM_FIELDS_ORDER:
                            st.session_state.active_form_field = FORM_FIELDS_ORDER[0]
                            st.info(f"Ativando primeiro campo: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                        else:
                            st.warning("Não há campos definidos no formulário.")
                    comando_processado = True
                    st.session_state.last_transcription_segment = ""
                    st.rerun()

                if "campo anterior" in texto_transcrito_lower and not comando_processado:
                    if st.session_state.active_form_field:
                        current_index = FORM_FIELDS_ORDER.index(st.session_state.active_form_field)
                        if current_index > 0:
                            st.session_state.active_form_field = FORM_FIELDS_ORDER[current_index - 1]
                            st.info(f"Campo ativo alterado para: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                        else:
                            st.info("Você já está no primeiro campo do formulário.")
                    else:
                        if FORM_FIELDS_ORDER:
                            st.session_state.active_form_field = FORM_FIELDS_ORDER[-1]
                            st.info(f"Ativando último campo: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                        else:
                            st.warning("Não há campos definidos no formulário.")
                    comando_processado = True
                    st.session_state.last_transcription_segment = ""
                    st.rerun()

                if ("parar preenchimento" in texto_transcrito_lower or
                    "sair do campo" in texto_transcrito_lower) and not comando_processado:
                    if st.session_state.active_form_field:
                        st.info(f"Saindo do campo **{st.session_state.active_form_field.replace('_', ' ').title()}**.")
                        st.session_state.active_form_field = None
                        st.session_state.last_transcription_segment = ""
                        st.rerun()
                        comando_processado = True
                    else:
                        st.warning("Nenhum campo específico está ativo para sair.")
                        comando_processado = True

                if not comando_processado and st.session_state.listening_active:
                    if st.session_state.active_form_field:
                        st.session_state[st.session_state.active_form_field] += " " + texto_transcrito_segmento
                    else:
                        st.session_state.transcricao_geral += " " + texto_transcrito_segmento

                self.buffer = b""

            return frame

    # --- UI Principal do Aplicativo (visível após o login) ---

    st.title("🗣️ Ficha de Atendimento de Fisioterapia")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Opções de Ficha")

        st.subheader("Fichas Padrão")
        for ficha_name, file_path in st.session_state.fichas_pdf_paths.items():
            if st.button(f"Abrir {ficha_name.title()} (PDF)", key=f"btn_open_pdf_{ficha_name}"):
                # Cache e carregamento do texto
                if file_path not in st.session_state.fichas_pdf_content_cache:
                    st.session_state.fichas_pdf_content_cache[file_path] = read_pdf_text(file_path)
                # st.session_state.transcricao_geral = st.session_state.fichas_pdf_content_cache[file_path] # AGORA COMEÇA VAZIO

                # Cache e carregamento das imagens do PDF
                if file_path not in st.session_state.fichas_pdf_images_cache:
                    st.session_state.fichas_pdf_images_cache[file_path] = get_pdf_images(file_path)
                st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[file_path]

                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = ficha_name
                st.session_state.transcricao_geral = "" # Zera para as respostas
                for key in FORM_FIELDS_MAP.values():
                    st.session_state[key] = ""
                st.session_state.active_form_field = None
                st.success(f"Ficha padrão '{ficha_name.title()}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                st.rerun()

        st.subheader("Nova Ficha")
        nova_ficha_tipo = st.text_input("Nome da Nova Ficha (Ex: Avaliação Postural)")
        if st.button("Criar Nova Ficha", key="btn_new_ficha"):
            if nova_ficha_tipo:
                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = f"Nova: {nova_ficha_tipo.strip()}"
                st.session_state.transcricao_geral = ""
                st.session_state.current_pdf_images = [] # Limpa imagens de PDF para nova ficha
                for key in FORM_FIELDS_MAP.values():
                    st.session_state[key] = ""
                st.session_state.active_form_field = FORM_FIELDS_ORDER[0] if FORM_FIELDS_ORDER else None
                st.success(f"Nova ficha '{nova_ficha_tipo.title()}' criada!")
                st.rerun()
            else:
                st.warning("Por favor, digite o nome da nova ficha.")

        st.subheader("Fichas de Pacientes Existentes")
        paciente_selecionado_ui = st.selectbox(
            "Selecione um Paciente",
            [""] + list(st.session_state.pacientes.keys()),
            key="select_paciente"
        )
        if paciente_selecionado_ui:
            fichas_do_paciente = st.session_state.pacientes[paciente_selecionado_ui]
            ficha_paciente_selecionada = st.selectbox(
                f"Selecione a Ficha para {paciente_selecionado_ui.title()}",
                [""] + list(fichas_do_paciente.keys()),
                key="select_ficha_paciente"
            )
            if st.button(f"Abrir Ficha de {paciente_selecionado_ui.title()}", key="btn_open_paciente_ficha"):
                if ficha_paciente_selecionada:
                    st.session_state.paciente_atual = paciente_selecionado_ui
                    st.session_state.tipo_ficha_aberta = ficha_paciente_selecionada
                    st.session_state.transcricao_geral = st.session_state.pacientes[paciente_selecionado_ui][ficha_paciente_selecionada]
                    st.session_state.current_pdf_images = [] # Limpa imagens de PDF ao abrir ficha de paciente
                    for key in FORM_FIELDS_MAP.values():
                        st.session_state[key] = ""
                    st.session_state.active_form_field = None
                    st.success(f"Ficha '{ficha_paciente_selecionada.title()}' do paciente '{paciente_selecionado_ui.title()}' aberta e texto carregado!")
                    st.rerun()
                else:
                    st.warning("Por favor, selecione uma ficha para o paciente.")

        st.markdown("---")
        st.header("Controle de Microfone")

        if st.session_state.listening_active:
            if st.button("Pausar Anotação de Voz ⏸️", key="btn_pause_listening"):
                st.session_state.listening_active = False
                st.info("Anotação de voz pausada. Diga 'retomar anotação' para continuar.")
                st.rerun()
        else:
            if st.button("Retomar Anotação de Voz ▶️", key="btn_resume_listening"):
                st.session_state.listening_active = True
                st.info("Anotação de voz retomada.")
                st.rerun()
        
        st.markdown(st.session_state.mic_status_message)
        if not st.session_state.listening_active:
            st.warning("Microfone em pausa. Comandos de voz para campos e fichas ainda funcionam.")
        
        st.markdown("---")

    with col2:
        st.header("Conteúdo da Ficha")

        if st.session_state.tipo_ficha_aberta:
            ficha_titulo = st.session_state.tipo_ficha_aberta.title()
            if st.session_state.paciente_atual:
                st.subheader(f"Ficha: {ficha_titulo} (Paciente: {st.session_state.paciente_atual.title()})")
            else:
                st.subheader(f"Ficha: {ficha_titulo}")
        else:
            st.subheader("Nenhuma ficha aberta")

        if st.session_state.last_transcription_segment:
            st.markdown(f"<p style='color: grey; font-size: 0.9em;'><i>Última transcrição: \"{st.session_state.last_transcription_segment}\"</i></p>", unsafe_allow_html=True)

        if st.session_state.active_form_field:
            friendly_name_active = next((k for k, v in FORM_FIELDS_MAP.items() if v == st.session_state.active_form_field), st.session_state.active_form_field)
            st.info(f"Ditando em: **{friendly_name_active.replace('_', ' ').title()}**")

        # NOVO: Exibe as imagens do PDF se houver alguma ficha PDF aberta
        if st.session_state.current_pdf_images:
            st.subheader("Visualização da Ficha (Guia)")
            for i, img in enumerate(st.session_state.current_pdf_images):
                st.image(img, caption=f"Página {i+1} do PDF", use_column_width=True)
            st.markdown("---") # Separador visual

        # Os campos específicos do formulário e o campo geral abaixo das imagens do PDF
        for friendly_name_display, field_key in FORM_FIELDS_MAP.items():
            st.text_area(
                f"{friendly_name_display.title()}:",
                value=st.session_state[field_key],
                key=field_key,
                height=150,
                help=f"Diga 'preencher {friendly_name_display}' para ativar este campo.",
                disabled=(st.session_state.active_form_field != field_key)
            )

        st.markdown("---")
        st.subheader("Observações Gerais da Ficha")
        st.text_area(
            "Texto da Ficha (Geral):",
            value=st.session_state.transcricao_geral,
            key="transcricao_geral_text_area",
            height=300,
            help="Texto ditado sem um campo específico ativo ou carregado de uma ficha existente.",
            disabled=(st.session_state.active_form_field is not None)
        )

        if st.button("Salvar Ficha (Simulação)", key="btn_save_ficha"):
            st.success("Ficha salva com sucesso! (Esta é uma simulação, os dados não são persistidos).")

    webrtc_ctx = webrtc_streamer(
        key="fisioterapia_voice_assistant",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.session_state.mic_status_message = "🟢 Microfone Conectado (Escutando)"
    else:
        st.session_state.mic_status_message = "🔴 Microfone Desconectado (Aguardando Conexão)"
