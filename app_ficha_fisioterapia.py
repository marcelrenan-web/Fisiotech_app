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

# --- REMOVIDOS OS CAMPOS FIXOS ---
# Não há mais FORM_FIELDS_MAP e FORM_FIELDS_ORDER aqui.
# A ideia é usar apenas a área de observações gerais para a transcrição.

# --- Caminhos para Armazenamento ---
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
            st.warning(f"Erro ao ler {UPLOADED_TEMPLATES_INDEX_FILE}. Criando um novo.")
            return {}
    return {}

def save_uploaded_templates_index(index_data):
    with open(UPLOADED_TEMPLATES_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4, ensure_ascii=False)

# --- Inicialização de Estados da Sessão Streamlit ---
if "logado" not in st.session_state:
    st.session_state.logado = False

if "fichas_padrao_paths" not in st.session_state:
    st.session_state.fichas_padrao_paths = {}

if "uploaded_fichas_data" not in st.session_state:
    st.session_state.uploaded_fichas_data = load_uploaded_templates_index()

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

# st.session_state.active_form_field não é mais necessário, pois não há campos específicos
# if "active_form_field" not in st.session_state:
#     st.session_state.active_form_field = None

if "listening_active" not in st.session_state:
    st.session_state.listening_active = True

if "mic_status_message" not in st.session_state:
    st.session_state.mic_status_message = "🔴 Microfone Desconectado"

if "webrtc_initialized" not in st.session_state:
    st.session_state.webrtc_initialized = False

# REMOVIDO o loop de inicialização de st.session_state[key] para campos fixos
# for key in FORM_FIELDS_MAP.values():
#     if key not in st.session_state:
#         st.session_state[key] = ""

# --- Funções Auxiliares ---

@st.cache_data(show_spinner="Extraindo texto do PDF...")
def read_pdf_text(file_path):
    if not os.path.exists(file_path):
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
                    st.rerun() # Adicionado rerun para refletir o status de escuta imediatamente
                elif "retomar anotação" in texto_transcrito_lower:
                    st.session_state.listening_active = True
                    st.session_state.last_transcription_segment = ""
                    comando_processado = True
                    st.rerun() # Adicionado rerun para refletir o status de escuta imediatamente

                # Lógica para abrir PDF via comando de voz (PADRÃO E UPLOADED)
                match_abrir_ficha_padrao_ou_upload = re.search(r"(?:abrir|mostrar) ficha de (.+)", texto_transcrito_lower)
                if match_abrir_ficha_padrao_ou_upload and not comando_processado:
                    ficha_solicitada = match_abrir_ficha_padrao_ou_upload.group(1).strip()
                    file_path_to_open = None

                    if ficha_solicitada in st.session_state.fichas_padrao_paths:
                        file_path_to_open = st.session_state.fichas_padrao_paths[ficha_solicitada]
                    elif ficha_solicitada in st.session_state.uploaded_fichas_data:
                        file_path_to_open = st.session_state.uploaded_fichas_data[ficha_solicitada]["path"]
                    
                    if file_path_to_open:
                        if file_path_to_open not in st.session_state.fichas_pdf_images_cache:
                            st.session_state.fichas_pdf_images_cache[file_path_to_open] = get_pdf_images(file_path_to_open)
                        st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[file_path_to_open]

                        st.session_state.paciente_atual = None
                        st.session_state.tipo_ficha_aberta = ficha_solicitada
                        st.session_state.transcricao_geral = ""
                        
                        # REMOVIDO o loop para resetar campos específicos
                        # for key in FORM_FIELDS_MAP.values():
                        #     st.session_state[key] = "" 
                        
                        # st.session_state.active_form_field = None # Não é mais necessário
                        st.success(f"Ficha '{ficha_solicitada.title()}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                        st.rerun()
                        comando_processado = True
                    else:
                        st.warning(f"Comando de ficha '{ficha_solicitada}' não reconhecido.")

                # Lógica para abrir paciente
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
                            
                            st.session_state.current_pdf_images = []
                            
                            # REMOVIDO o loop para resetar campos específicos
                            # for key in FORM_FIELDS_MAP.values():
                            #     st.session_state[key] = ""
                            
                            # st.session_state.active_form_field = None # Não é mais necessário
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
                    
                    st.session_state.current_pdf_images = []
                    
                    # REMOVIDO o loop para resetar campos específicos
                    # for key in FORM_FIELDS_MAP.values():
                    #     st.session_state[key] = ""
                    
                    # st.session_state.active_form_field = None # Não é mais necessário
                    st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Dite em observações gerais.")
                    st.rerun()
                    comando_processado = True

                # REMOVIDA A LÓGICA DE ATIVAÇÃO DE CAMPO POR VOZ
                # match_preencher_campo = re.search(r"preencher (.+)", texto_transcrito_lower)
                # if match_preencher_campo and not comando_processado:
                #     campo_falado = match_preencher_campo.group(1).strip()
                #     # ... (lógica de busca e ativação de campo que não usaremos mais)

                # REMOVIDA A NAVEGAÇÃO ENTRE CAMPOS
                # if "proximo campo" in texto_transcrito_lower and not comando_processado:
                #     # ... (lógica de próximo campo)
                # if "campo anterior" in texto_transcrito_lower and not comando_processado:
                #     # ... (lógica de campo anterior)

                # REMOVIDA A SAÍDA DE CAMPO
                # if ("parar preenchimento" in texto_transcrito_lower or
                #     "sair do campo" in texto_transcrito_lower) and not comando_processado:
                #     # ... (lógica de sair do campo)

                if not comando_processado and st.session_state.listening_active:
                    # Agora, toda a transcrição vai para o campo geral
                    st.session_state.transcricao_geral += " " + texto_transcrito_segmento
                    # st.rerun() # Adicionado rerun para atualizar o texto imediatamente (pode causar mais reruns)


                self.buffer = b""

            return frame

    # --- UI Principal do Aplicativo (visível após o login) ---

    st.title("🗣️ Ficha de Atendimento de Fisioterapia")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Opções de Ficha")

        # --- SEÇÃO DE UPLOAD DE FICHAS MODELO ---
        st.subheader("Upload de Nova Ficha Modelo (PDF)")
        uploaded_file = st.file_uploader("Escolha um arquivo PDF para upload", type="pdf")
        new_ficha_name = st.text_input("Nome para esta nova ficha modelo (Ex: 'Ficha de Coluna')", key="new_uploaded_ficha_name")
        
        if uploaded_file is not None and new_ficha_name:
            if st.button("Salvar Ficha Modelo Uploaded", key="btn_save_uploaded_template"):
                sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '', new_ficha_name.lower().replace(" ", "_"))
                file_ext = os.path.splitext(uploaded_file.name)[1]
                save_path = os.path.join(UPLOADED_TEMPLATES_DIR, f"{sanitized_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}")

                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.uploaded_fichas_data[new_ficha_name.lower()] = {
                        "name": new_ficha_name,
                        "path": save_path
                    }
                    save_uploaded_templates_index(st.session_state.uploaded_fichas_data)
                    st.success(f"Ficha modelo '{new_ficha_name}' salva e pronta para uso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao salvar o arquivo: {e}")
        elif uploaded_file is None and new_ficha_name:
            st.warning("Por favor, selecione um arquivo PDF antes de salvar a ficha modelo.")

        st.markdown("---")

        # --- SELEÇÃO DE FICHAS MODELO (Padrão e Uploadadas) ---
        st.subheader("Abrir Ficha Modelo (PDF)")
        
        all_template_fichas = {}
        for name, path in st.session_state.fichas_padrao_paths.items():
            if os.path.exists(path):
                all_template_fichas[name.lower()] = {"name": name, "path": path}
        
        keys_to_remove = []
        for key, info in st.session_state.uploaded_fichas_data.items():
            if os.path.exists(info['path']):
                all_template_fichas[key] = info
            else:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            st.warning(f"Ficha Modelo '{st.session_state.uploaded_fichas_data[key]['name']}' não encontrada em '{st.session_state.uploaded_fichas_data[key]['path']}'. Será removida da lista.")
            del st.session_state.uploaded_fichas_data[key]
        if keys_to_remove:
            save_uploaded_templates_index(st.session_state.uploaded_fichas_data)
            st.rerun()
        
        template_ficha_options = [""] + sorted([info["name"].title() for info in all_template_fichas.values()])
        selected_template_ficha_name = st.selectbox(
            "Selecione uma ficha modelo para abrir:",
            template_ficha_options,
            key="select_template_ficha"
        )

        if selected_template_ficha_name and st.button(f"Abrir Ficha Modelo Selecionada", key="btn_open_selected_template"):
            selected_ficha_path = None
            for key, info in all_template_fichas.items():
                if info["name"].lower() == selected_template_ficha_name.lower():
                    selected_ficha_path = info["path"]
                    break
            
            if selected_ficha_path:
                if selected_ficha_path not in st.session_state.fichas_pdf_images_cache:
                    st.session_state.fichas_pdf_images_cache[selected_ficha_path] = get_pdf_images(selected_ficha_path)
                st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[selected_ficha_path]

                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = selected_template_ficha_name.lower()
                st.session_state.transcricao_geral = ""
                # REMOVIDO o reset dos campos fixos
                # for key in FORM_FIELDS_MAP.values():
                #     st.session_state[key] = ""
                # st.session_state.active_form_field = None # Não é mais necessário
                st.success(f"Ficha modelo '{selected_template_ficha_name}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                st.rerun()
            else:
                st.error("Erro ao encontrar o caminho da ficha selecionada.")

        # --- Opção para deletar fichas salvas ---
        if st.checkbox("Gerenciar Fichas Modelos Salvas (Deletar)"):
            if st.session_state.uploaded_fichas_data:
                ficha_keys_to_delete = list(st.session_state.uploaded_fichas_data.keys())
                ficha_to_delete_name_display = st.selectbox(
                    "Selecione uma ficha para deletar:", 
                    [""] + [st.session_state.uploaded_fichas_data[k]['name'] for k in ficha_keys_to_delete],
                    key="delete_uploaded_ficha_select_name"
                )
                
                if ficha_to_delete_name_display:
                    original_key_to_delete = next((k for k, v in st.session_state.uploaded_fichas_data.items() if v['name'] == ficha_to_delete_name_display), None)
                    if original_key_to_delete and st.button(f"Deletar '{ficha_to_delete_name_display}'", key="btn_delete_uploaded_ficha"):
                        file_path_to_delete = st.session_state.uploaded_fichas_data[original_key_to_delete]['path']
                        
                        if os.path.exists(file_path_to_delete):
                            os.remove(file_path_to_delete)
                            st.info(f"Arquivo '{os.path.basename(file_path_to_delete)}' deletado do disco.")
                        else:
                            st.warning(f"Arquivo '{os.path.basename(file_path_to_delete)}' não encontrado no disco (já pode ter sido deletado).")
                        
                        del st.session_state.uploaded_fichas_data[original_key_to_delete]
                        save_uploaded_templates_index(st.session_state.uploaded_fichas_data)
                        
                        if file_path_to_delete in st.session_state.fichas_pdf_content_cache:
                            del st.session_state.fichas_pdf_content_cache[file_path_to_delete]
                        if file_path_to_delete in st.session_state.fichas_pdf_images_cache:
                            del st.session_state.fichas_pdf_images_cache[file_path_to_delete]
                        
                        st.success(f"Ficha modelo '{ficha_to_delete_name_display}' deletada do aplicativo.")
                        st.rerun()
            else:
                st.info("Nenhuma ficha modelo para deletar.")


        st.markdown("---")
        st.subheader("Nova Ficha em Branco")
        nova_ficha_tipo = st.text_input("Nome da Nova Ficha (Ex: Avaliação Postural)", key="new_blank_ficha_name")
        if st.button("Criar Nova Ficha em Branco", key="btn_new_blank_ficha"):
            if nova_ficha_tipo:
                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = f"Nova: {nova_ficha_tipo.strip()}"
                st.session_state.transcricao_geral = ""
                st.session_state.current_pdf_images = []
                # REMOVIDO o reset dos campos fixos
                # for key in FORM_FIELDS_MAP.values():
                #     st.session_state[key] = ""
                # st.session_state.active_form_field = None # Não é mais necessário
                st.success(f"Nova ficha '{nova_ficha_tipo.title()}' criada! Comece a ditar em observações gerais.")
                st.rerun()
            else:
                st.warning("Por favor, digite o nome da nova ficha.")

        st.subheader("Fichas de Pacientes Existentes (Simulado)")
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
                    st.session_state.current_pdf_images = []
                    # REMOVIDO o reset dos campos fixos
                    # for key in FORM_FIELDS_MAP.values():
                    #     st.session_state[key] = ""
                    # st.session_state.active_form_field = None # Não é mais necessário
                    st.success(f"Ficha '{ficha_paciente_selecionada.title()}' do paciente '{paciente_selecionado_ui.title()}' aberta e texto carregado!")
                    st.rerun()
                else:
                    st.warning("Por favor, selecione uma ficha para o paciente.")

        st.markdown("---")
        st.header("Controle de Microfone")

        # Configuração do WebRTC
        # A chamada a webrtc_streamer precisa ocorrer a cada rerun para o componente funcionar.
        # No entanto, a lógica de inicialização de estados internos (como webrtc_initialized)
        # deve ser controlada para evitar chamadas duplicadas ou indesejadas.
        webrtc_ctx = webrtc_streamer(
            key="audio_recorder_streamer",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
        )
        
        # Atualiza o status do microfone em cada rerun baseado no webrtc_ctx
        if webrtc_ctx.state.playing:
            st.session_state.mic_status_message = "🟢 Microfone Conectado (Ouvindo)"
        else:
            st.session_state.mic_status_message = "🔴 Microfone Desconectado"


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

        # A mensagem de "Ditando em:" não é mais necessária, pois não há campos específicos
        # if st.session_state.active_form_field:
        #     friendly_name_active = next((k for k, v in FORM_FIELDS_MAP.items() if v == st.session_state.active_form_field), st.session_state.active_form_field)
        #     st.info(f"Ditando em: **{friendly_name_active.replace('_', ' ').title()}**")

        # Exibe as imagens do PDF se houver alguma ficha PDF aberta
        if st.session_state.current_pdf_images:
            st.subheader("Visualização da Ficha (Guia)")
            for i, img in enumerate(st.session_state.current_pdf_images):
                st.image(img, caption=f"Página {i+1} do PDF", use_column_width=True)
            st.markdown("---")

        # REMOVIDO o loop para renderizar campos específicos
        # for friendly_name_display, field_key in FORM_FIELDS_MAP.items():
        #     st.text_area(
        #         f"{friendly_name_display.title()}:",
        #         value=st.session_state[field_key],
        #         key=field_key,
        #         height=150,
        #         help=f"Diga 'preencher {friendly_name_display}' para ativar este campo.",
        #         disabled=(st.session_state.active_form_field != field_key)
        #     )

        st.markdown("---")
        st.subheader("Observações Gerais da Ficha")
        st.text_area(
            "Texto da Ficha (Geral):",
            value=st.session_state.transcricao_geral,
            key="transcricao_geral",
            height=300,
            # disabled=(st.session_state.active_form_field is not None) # Não é mais necessário, sempre habilitado
        )

        st.markdown("---")
        if st.session_state.tipo_ficha_aberta:
            if st.button("Salvar Ficha", key="btn_save_ficha"):
                if st.session_state.paciente_atual:
                    # Atualiza a ficha existente do paciente
                    st.session_state.pacientes[st.session_state.paciente_atual][st.session_state.tipo_ficha_aberta] = st.session_state.transcricao_geral
                    st.success(f"Ficha de '{st.session_state.tipo_ficha_aberta.title()}' do paciente '{st.session_state.paciente_atual.title()}' atualizada com sucesso!")
                elif st.session_state.tipo_ficha_aberta.startswith("Nova:"):
                    # Salva como nova ficha para um NOVO paciente (ou adiciona a um existente)
                    new_patient_name = st.text_input("Nome do Paciente para Salvar:", key="new_patient_name_save_on_save_button")
                    if new_patient_name:
                        patient_key = new_patient_name.lower().strip()
                        if patient_key not in st.session_state.pacientes:
                            st.session_state.pacientes[patient_key] = {}
                        
                        ficha_name_to_save = st.session_state.tipo_ficha_aberta.replace("Nova: ", "").strip()
                        st.session_state.pacientes[patient_key][ficha_name_to_save.lower()] = st.session_state.transcricao_geral
                        st.success(f"Nova ficha '{ficha_name_to_save.title()}' salva para o paciente '{patient_key.title()}'!")
                        st.session_state.paciente_atual = patient_key
                        st.session_state.tipo_ficha_aberta = ficha_name_to_save.lower()
                        st.rerun()
                    else:
                        st.warning("Por favor, digite o nome do paciente para salvar a nova ficha.")
                else:
                    st.warning("Não é possível salvar a ficha. Abra uma ficha existente de paciente ou crie uma nova ficha em branco.")
        else:
            st.info("Abra ou crie uma ficha para começar a ditar.")
