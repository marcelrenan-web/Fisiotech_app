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

# --- Caminhos para Armazenamento ---
UPLOADED_TEMPLATES_DIR = "dados/uploaded_fichas_templates"
UPLOADED_TEMPLATES_INDEX_FILE = "dados/uploaded_fichas_index.json"
SAVED_RECORDS_DIR = "dados/saved_patient_records" # Para fichas preenchidas
PATIENT_RECORDS_FILE = "dados/patient_records.json" # Novo arquivo para persistir dados de pacientes

# Garante que os diretórios existam
os.makedirs(UPLOADED_TEMPLATES_DIR, exist_ok=True)
os.makedirs(SAVED_RECORDS_DIR, exist_ok=True) # Pode ser usado para salvar fichas como arquivos individuais no futuro

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

# --- Funções para Persistência de Dados de Pacientes ---
def load_patient_records():
    if os.path.exists(PATIENT_RECORDS_FILE):
        try:
            with open(PATIENT_RECORDS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Erro ao ler {PATIENT_RECORDS_FILE}. Iniciando com dados vazios.")
            return {}
    return {}

def save_patient_records(records_data):
    with open(PATIENT_RECORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(records_data, f, indent=4, ensure_ascii=False)

# --- Inicialização de Estados da Sessão Streamlit ---
if "logado" not in st.session_state:
    st.session_state.logado = False

if "fichas_padrao_paths" not in st.session_state:
    # Por enquanto, mantemos vazio ou adicionamos uma ficha padrão se existir um PDF em "dados/"
    # Ex: st.session_state.fichas_padrao_paths = {"ficha de avaliação id fisiopuntura": "dados/ficha_avaliacao_id_fisiopuntura.pdf"}
    # CERTIFIQUE-SE DE TER ESSE ARQUIVO NO DIRETÓRIO `dados/`
    st.session_state.fichas_padrao_paths = {
        "ficha de avaliação id fisiopuntura": "dados/FICHA_DE_AVALIAÇÃO_ID_FISIOPUNTURA.pdf"
    }


if "uploaded_fichas_data" not in st.session_state:
    st.session_state.uploaded_fichas_data = load_uploaded_templates_index()

if "fichas_pdf_content_cache" not in st.session_state:
    st.session_state.fichas_pdf_content_cache = {}
if "fichas_pdf_images_cache" not in st.session_state:
    st.session_state.fichas_pdf_images_cache = {}

if "pacientes" not in st.session_state:
    st.session_state.pacientes = load_patient_records() # Carrega os registros de pacientes

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

if "listening_active" not in st.session_state:
    st.session_state.listening_active = True

if "mic_status_message" not in st.session_state:
    st.session_state.mic_status_message = "🔴 Microfone Desconectado"

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
            # Recomendo usar um modelo multilíngue maior ou um específico para pt
            # "base" é multilíngue, "small" é multilíngue e mais preciso.
            model = whisper.load_model("base") # Ou "small", "medium"
            st.success("Modelo Whisper 'base' carregado!")
        except Exception as e:
            st.error(f"Erro ao carregar modelo Whisper: {e}. Verifique sua conexão ou instalação.")
            st.stop() # Interrompe a execução se o modelo não carregar
        return model

    model = carregar_modelo()

    def corrigir_termos(texto):
        # Esta função pode ser mais útil com um modelo multilíngue ou se houver tendências de erro específicas
        correcoes = {
            "tendinite": "tendinite",
            "cervicalgia": "cervicalgia",
            "lombar": "região lombar",
            "reabilitação funcional": "reabilitação funcional",
            "fisioterapia do ombro": "fisioterapia de ombro",
            "dor nas costas": "algia na coluna",
            # Adicione mais correções conforme necessário
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

            if len(self.buffer) > 32000 * 5: # Processa a cada 5 segundos de áudio
                audio_np = np.frombuffer(self.buffer, np.float32)
                audio_np = whisper.pad_or_trim(audio_np)
                # Ensure the model is on the correct device (CPU or CUDA)
                mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
                options = whisper.DecodingOptions(language="pt", fp16=False) # Especifica português
                result = whisper.decode(model, mel, options)
                
                texto_transcrito_segmento = corrigir_termos(result.text).strip()
                st.session_state.last_transcription_segment = texto_transcrito_segmento
                
                comando_processado = False
                texto_transcrito_lower = texto_transcrito_segmento.lower()

                # Comandos de controle de escuta
                if "pausar anotação" in texto_transcrito_lower:
                    st.session_state.listening_active = False
                    st.session_state.last_transcription_segment = "" # Limpa para não ficar exibindo o comando
                    comando_processado = True
                    # st.rerun() # Removido para evitar reruns excessivos no AudioProcessor
                elif "retomar anotação" in texto_transcrito_lower:
                    st.session_state.listening_active = True
                    st.session_state.last_transcription_segment = "" # Limpa para não ficar exibindo o comando
                    comando_processado = True
                    # st.rerun() # Removido para evitar reruns excessivos no AudioProcessor

                # Lógica para abrir PDF via comando de voz (PADRÃO E UPLOADED)
                match_abrir_ficha_padrao_ou_upload = re.search(r"(?:abrir|mostrar) ficha de (.+)", texto_transcrito_lower)
                if match_abrir_ficha_padrao_ou_upload and not comando_processado:
                    ficha_solicitada = match_abrir_ficha_padrao_ou_upload.group(1).strip()
                    file_path_to_open = None

                    # Prioriza fichas uploadadas, depois as padrão
                    if ficha_solicitada in st.session_state.uploaded_fichas_data:
                        file_path_to_open = st.session_state.uploaded_fichas_data[ficha_solicitada]["path"]
                    elif ficha_solicitada in st.session_state.fichas_padrao_paths:
                        file_path_to_open = st.session_state.fichas_padrao_paths[ficha_solicitada]
                    
                    if file_path_to_open:
                        if file_path_to_open not in st.session_state.fichas_pdf_images_cache:
                            st.session_state.fichas_pdf_images_cache[file_path_to_open] = get_pdf_images(file_path_to_open)
                        st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[file_path_to_open]

                        st.session_state.paciente_atual = None
                        st.session_state.tipo_ficha_aberta = ficha_solicitada
                        st.session_state.transcricao_geral = ""
                        st.success(f"Ficha '{ficha_solicitada.title()}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                        comando_processado = True
                        st.rerun() # Rerun aqui é importante para atualizar a UI imediatamente com a ficha
                    else:
                        st.warning(f"Comando de ficha '{ficha_solicitada}' não reconhecido.")
                        comando_processado = True # Marca como processado para não adicionar ao texto geral

                # Lógica para abrir ficha de paciente existente
                match_abrir_paciente_ficha = re.search(r"abrir ficha do paciente (.+?) (?:de|da)? (.+)", texto_transcrito_lower)
                if match_abrir_paciente_ficha and not comando_processado:
                    nome_paciente_falado = match_abrir_paciente_ficha.group(1).strip()
                    tipo_ficha_falado = match_abrir_paciente_ficha.group(2).strip()
                    
                    found_patient = None
                    for p_name_db in st.session_state.pacientes:
                        if nome_paciente_falado in p_name_db: # Busca parcial
                            found_patient = p_name_db
                            break

                    if found_patient:
                        if tipo_ficha_falado in st.session_state.pacientes[found_patient]:
                            st.session_state.paciente_atual = found_patient
                            st.session_state.tipo_ficha_aberta = tipo_ficha_falado
                            st.session_state.transcricao_geral = st.session_state.pacientes[found_patient][tipo_ficha_falado]
                            
                            st.session_state.current_pdf_images = [] # Limpa visualização de PDF
                            
                            st.success(f"Ficha '{tipo_ficha_falado.title()}' do paciente '{found_patient.title()}' aberta e texto carregado!")
                            comando_processado = True
                            st.rerun() # Rerun aqui é importante para atualizar a UI
                        else:
                            st.warning(f"Não foi possível encontrar a ficha '{tipo_ficha_falado.title()}' para o paciente '{found_patient.title()}'.")
                            comando_processado = True
                    else:
                        st.warning(f"Paciente '{nome_paciente_falado.title()}' não encontrado.")
                        comando_processado = True

                # Lógica para criar nova ficha em branco
                match_nova_ficha = re.search(r"nova ficha de (.+)", texto_transcrito_lower)
                if match_nova_ficha and not comando_processado:
                    tipo_nova_ficha = match_nova_ficha.group(1).strip()
                    st.session_state.paciente_atual = None
                    st.session_state.tipo_ficha_aberta = f"Nova: {tipo_nova_ficha}"
                    st.session_state.transcricao_geral = ""
                    
                    st.session_state.current_pdf_images = [] # Limpa visualização de PDF
                    
                    st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Dite em observações gerais.")
                    comando_processado = True
                    st.rerun() # Rerun aqui para atualizar a UI

                if not comando_processado and st.session_state.listening_active:
                    # Agora, toda a transcrição vai para o campo geral
                    # Adiciona apenas se o segmento não estiver vazio para evitar espaços duplos
                    if texto_transcrito_segmento:
                        st.session_state.transcricao_geral += " " + texto_transcrito_segmento
                
                self.buffer = b"" # Limpa o buffer após processar

            return frame

    # --- UI Principal do Aplicativo (visível após o login) ---

    st.title("🗣️ Ficha de Atendimento de Fisioterapia")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Opções de Ficha")

        # --- SEÇÃO DE UPLOAD DE FICHAS MODELO ---
        st.subheader("Upload de Nova Ficha Modelo (PDF)")
        uploaded_file = st.file_uploader("Escolha um arquivo PDF para upload", type="pdf", key="file_uploader_template")
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
                    # Limpa os campos após o salvamento
                    st.session_state.new_uploaded_ficha_name = "" 
                    # uploaded_file não pode ser limpo diretamente via session_state, um rerun resolve
                    st.rerun() 
                except Exception as e:
                    st.error(f"Erro ao salvar o arquivo: {e}")
        elif uploaded_file is None and new_ficha_name:
            st.warning("Por favor, selecione um arquivo PDF antes de salvar a ficha modelo.")

        st.markdown("---")

        # --- SELEÇÃO DE FICHAS MODELO (Padrão e Uploadadas) ---
        st.subheader("Abrir Ficha Modelo (PDF)")
        
        all_template_fichas = {}
        # Adiciona fichas padrão
        for name, path in st.session_state.fichas_padrao_paths.items():
            if os.path.exists(path):
                all_template_fichas[name.lower()] = {"name": name, "path": path}
            else:
                st.warning(f"Ficha Padrão '{name}' não encontrada em '{path}'. Verifique o diretório 'dados/'.")
        
        # Adiciona fichas uploadadas e remove as não encontradas
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
            st.rerun() # Rerun para atualizar a lista após remoção
            
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
                        
                        # Limpa cache se a ficha deletada estava no cache
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
        nova_ficha_tipo = st.text_input("Nome da Nova Ficha (Ex: Avaliação Postural)", key="new_blank_ficha_name_input")
        if st.button("Criar Nova Ficha em Branco", key="btn_new_blank_ficha"):
            if nova_ficha_tipo:
                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = f"Nova: {nova_ficha_tipo.strip()}"
                st.session_state.transcricao_geral = ""
                st.session_state.current_pdf_images = []
                st.success(f"Nova ficha '{nova_ficha_tipo.title()}' criada! Comece a ditar em observações gerais.")
                st.rerun()
            else:
                st.warning("Por favor, digite o nome da nova ficha.")

        st.subheader("Fichas de Pacientes Existentes")
        # Mostra a lista de pacientes no selectbox, com opção para "Novo Paciente"
        all_patients_keys = list(st.session_state.pacientes.keys())
        paciente_selecionado_ui = st.selectbox(
            "Selecione um Paciente ou digite um nome para um novo:",
            [""] + sorted(all_patients_keys) + ["-- Novo Paciente --"],
            key="select_paciente"
        )
        
        ficha_paciente_selecionada = None
        if paciente_selecionado_ui and paciente_selecionado_ui != "-- Novo Paciente --":
            fichas_do_paciente = st.session_state.pacientes.get(paciente_selecionado_ui, {})
            ficha_paciente_selecionada = st.selectbox(
                f"Selecione a Ficha para {paciente_selecionado_ui.title()}",
                [""] + list(fichas_do_paciente.keys()),
                key="select_ficha_paciente"
            )
        
        if st.button(f"Abrir Ficha de Paciente", key="btn_open_paciente_ficha"):
            if paciente_selecionado_ui == "-- Novo Paciente --":
                st.warning("Para 'Novo Paciente', use a opção 'Criar Nova Ficha em Branco' e salve-a com o nome do novo paciente.")
            elif paciente_selecionado_ui and ficha_paciente_selecionada:
                st.session_state.paciente_atual = paciente_selecionado_ui
                st.session_state.tipo_ficha_aberta = ficha_paciente_selecionada
                st.session_state.transcricao_geral = st.session_state.pacientes[paciente_selecionado_ui][ficha_paciente_selecionada]
                st.session_state.current_pdf_images = []
                st.success(f"Ficha '{ficha_paciente_selecionada.title()}' do paciente '{paciente_selecionado_ui.title()}' aberta e texto carregado!")
                st.rerun()
            else:
                st.warning("Por favor, selecione um paciente e uma ficha para abrir.")

        st.markdown("---")
        st.header("Controle de Microfone")

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
        
        if webrtc_ctx.state.playing:
            st.session_state.mic_status_message = "🟢 Microfone Conectado (Ouvindo)"
        else:
            st.session_state.mic_status_message = "🔴 Microfone Desconectado"

        if st.session_state.listening_active:
            if st.button("Pausar Anotação de Voz ⏸️", key="btn_pause_listening"):
                st.session_state.listening_active = False
                st.info("Anotação de voz pausada. Diga 'retomar anotação' para continuar.")
                # Nao precisa de rerun aqui, o AudioProcessor já lida com o estado.
        else:
            if st.button("Retomar Anotação de Voz ▶️", key="btn_resume_listening"):
                st.session_state.listening_active = True
                st.info("Anotação de voz retomada.")
                # Nao precisa de rerun aqui.
                
        st.markdown(st.session_state.mic_status_message)
        if not st.session_state.listening_active:
            st.warning("Microfone em pausa. Comandos de voz para abrir fichas ainda funcionam, mas o ditado geral está pausado.")
            
        st.markdown("---")

    with col2:
        st.header("Conteúdo da Ficha")

        if st.session_state.tipo_ficha_aberta:
            ficha_titulo = st.session_state.tipo_ficha_aberta.title()
            if st.session_state.paciente_atual:
                st.subheader(f"Ficha: {ficha_titulo} (Paciente: {st.session_state.paciente_atual.title()})")
            else:
                # Trata o caso de "Nova: Nome da Ficha"
                display_title = ficha_titulo.replace("Nova: ", "") if "nova:" in ficha_titulo.lower() else ficha_titulo
                st.subheader(f"Ficha: {display_title}")
        else:
            st.subheader("Nenhuma ficha aberta")

        if st.session_state.last_transcription_segment:
            st.markdown(f"<p style='color: grey; font-size: 0.9em;'><i>Última transcrição: \"{st.session_state.last_transcription_segment}\"</i></p>", unsafe_allow_html=True)

        # Exibe as imagens do PDF se houver alguma ficha PDF aberta
        if st.session_state.current_pdf_images:
            st.subheader("Visualização da Ficha (Guia PDF)")
            for i, img in enumerate(st.session_state.current_pdf_images):
                st.image(img, caption=f"Página {i+1} do PDF", use_column_width=True)
            st.markdown("---")

        st.subheader("Observações Gerais da Ficha")
        # Usamos o placeholder para mostrar ao usuário que o ditado ocorre aqui
        st.session_state.transcricao_geral = st.text_area(
            "Texto da Ficha (Geral):",
            value=st.session_state.transcricao_geral,
            key="transcricao_geral_text_area", # Renomeada para evitar conflito com session_state key
            height=300,
            help="Todo o ditado por voz, a menos que seja um comando, será inserido aqui."
        )

        st.markdown("---")
        if st.session_state.tipo_ficha_aberta:
            if st.button("Salvar Ficha", key="btn_save_ficha"):
                if st.session_state.paciente_atual:
                    # Atualiza a ficha existente do paciente
                    st.session_state.pacientes[st.session_state.paciente_atual][st.session_state.tipo_ficha_aberta] = st.session_state.transcricao_geral
                    save_patient_records(st.session_state.pacientes) # Salva no arquivo
                    st.success(f"Ficha de '{st.session_state.tipo_ficha_aberta.title()}' do paciente '{st.session_state.paciente_atual.title()}' atualizada com sucesso!")
                    st.rerun() # Rerun para limpar a mensagem de "Nome do Paciente para Salvar" se ela apareceu
                elif st.session_state.tipo_ficha_aberta.startswith("Nova:"):
                    # Salva como nova ficha para um NOVO paciente (ou adiciona a um existente)
                    new_patient_name_for_save = st.text_input("Nome do Paciente para Salvar a Nova Ficha:", key="new_patient_name_save_on_save_button")
                    if new_patient_name_for_save:
                        patient_key = new_patient_name_for_save.lower().strip()
                        if patient_key not in st.session_state.pacientes:
                            st.session_state.pacientes[patient_key] = {}
                        
                        ficha_name_to_save = st.session_state.tipo_ficha_aberta.replace("Nova: ", "").strip().lower()
                        st.session_state.pacientes[patient_key][ficha_name_to_save] = st.session_state.transcricao_geral
                        save_patient_records(st.session_state.pacientes) # Salva no arquivo
                        st.success(f"Nova ficha '{ficha_name_to_save.title()}' salva para o paciente '{patient_key.title()}'!")
                        st.session_state.paciente_atual = patient_key
                        st.session_state.tipo_ficha_aberta = ficha_name_to_save
                        st.rerun()
                    else:
                        st.warning("Por favor, digite o nome do paciente para salvar a nova ficha.")
                else:
                    st.warning("Não é possível salvar a ficha. Abra uma ficha existente de paciente ou crie uma nova ficha em branco.")
        else:
            st.info("Abra ou crie uma ficha para começar a ditar.")
