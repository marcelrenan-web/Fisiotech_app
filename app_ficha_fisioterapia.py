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
# O path continua aqui, mas a exibição na UI será via selectbox
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
                        if file_path_to_open not in st.session_state.fichas_pdf_content_cache:
                            st.session_state.fichas_pdf_content_cache[file_path_to_open] = read_pdf_text(file_path_to_open)
                        
                        if file_path_to_open not in st.session_state.fichas_pdf_images_cache:
                            st.session_state.fichas_pdf_images_cache[file_path_to_open] = get_pdf_images(file_path_to_open)
                        st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[file_path_to_open]

                        st.session_state.paciente_atual = None
                        st.session_state.tipo_ficha_aberta = ficha_solicitada
                        st.session_state.transcricao_geral = ""
                        
                        for key in FORM_FIELDS_MAP.values():
                            st.session_state[key] = "" 
                        
                        st.session_state.active_form_field = None
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
        
        # Combinar as listas de fichas padrão e uploadadas para o selectbox
        all_template_fichas = {}
        # Adiciona fichas padrão
        for name, path in st.session_state.fichas_padrao_paths.items():
            if os.path.exists(path):
                all_template_fichas[name.lower()] = {"name": name, "path": path}
            else:
                pass # Não exibe warning para fichas padrão ausentes na UI, apenas no código de inicialização
        
        # Adiciona fichas uploadadas, sobrescrevendo se houver conflito de nome
        # Também remove entradas de fichas uploadadas que não existem mais no disco
        keys_to_remove = []
        for key, info in st.session_state.uploaded_fichas_data.items():
            if os.path.exists(info['path']):
                all_template_fichas[key] = info # Key já está em lower()
            else:
                keys_to_remove.append(key) # Marca para remoção
        
        for key in keys_to_remove:
            st.warning(f"Ficha Modelo '{st.session_state.uploaded_fichas_data[key]['name']}' não encontrada em '{st.session_state.uploaded_fichas_data[key]['path']}'. Será removida da lista.")
            del st.session_state.uploaded_fichas_data[key]
        if keys_to_remove:
            save_uploaded_templates_index(st.session_state.uploaded_fichas_data)
            st.rerun() # Recarrega para refletir as remoções
        
        template_ficha_options = [""] + sorted([info["name"].title() for info in all_template_fichas.values()])
        selected_template_ficha_name = st.selectbox(
            "Selecione uma ficha modelo para abrir:",
            template_ficha_options,
            key="select_template_ficha"
        )

        if selected_template_ficha_name and st.button(f"Abrir Ficha Modelo", key="btn_open_selected_template"): # Botão mais genérico
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
                for key in FORM_FIELDS_MAP.values():
                    st.session_state[key] = ""
                st.session_state.active_form_field = None
                st.success(f"Ficha modelo '{selected_template_ficha_name}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                st.rerun()
            else:
                st.error("Erro ao encontrar o caminho da ficha selecionada.")

        # --- Opção para deletar fichas salvas (mantido como checkbox para ocultar/exibir) ---
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
                for key in FORM_FIELDS_MAP.values():
                    st.session_state[key] = ""
                st.session_state.active_form_field = FORM_FIELDS_ORDER[0] if FORM_FIELDS_ORDER else None
                st.success(f"Nova ficha '{nova_ficha_tipo.title()}' criada!")
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

        # Exibe as imagens do PDF se houver alguma ficha PDF aberta
        if st.session_state.current_pdf_images:
            st.subheader("Visualização da Ficha (Guia)")
            for i, img in enumerate(st.session_state.current_pdf_images):
                st.image(img, caption=f"Página {i+1} do PDF", use_column_width=True)
            st.markdown("---")

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

        # NOVO: Botão para salvar a ficha preenchida
        if st.button("Salvar Ficha Preenchida", key="btn_save_filled_ficha"):
            if st.session_state.tipo_ficha_aberta and (st.session_state.transcricao_geral.strip() or any(st.session_state[k].strip() for k in FORM_FIELDS_MAP.values())):
                
                base_name = st.session_state.paciente_atual.replace(' ', '_') if st.session_state.paciente_atual else 'NovaFicha'
                ficha_type_name = st.session_state.tipo_ficha_aberta.replace(' ', '_').replace(':', '')
                record_id = f"{base_name}_{ficha_type_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                record_path = os.path.join(SAVED_RECORDS_DIR, record_id)
                
                ficha_data = {
                    "tipo_ficha": st.session_state.tipo_ficha_aberta,
                    "paciente": st.session_state.paciente_atual,
                    "data_preenchimento": datetime.now().isoformat(),
                    "observacoes_gerais": st.session_state.transcricao_geral,
                    "campos_especificos": {friendly_name: st.session_state[field_key] for friendly_name, field_key in FORM_FIELDS_MAP.items()},
                    "modelo_pdf_usado_path": st.session_state.fichas_padrao_paths.get(st.session_state.tipo_ficha_aberta) 
                                            or (st.session_state.uploaded_fichas_data.get(st.session_state.tipo_ficha_aberta.lower()) and st.session_state.uploaded_fichas_data[st.session_state.tipo_ficha_aberta.lower()]["path"])
                }

                try:
                    with open(record_path, "w", encoding="utf-8") as f:
                        json.dump(ficha_data, f, indent=4, ensure_ascii=False)
                    st.success(f"Ficha preenchida salva como: {record_id}")
                except Exception as e:
                    st.error(f"Erro ao salvar a ficha preenchida: {e}")
            else:
                st.warning("Não há ficha aberta ou conteúdo para salvar. Por favor, preencha algo.")

        st.markdown("---")
        st.subheader("Acessar Fichas Preenchidas Salvas")
        
        saved_records = [f for f in os.listdir(SAVED_RECORDS_DIR) if f.endswith('.json')]
        if saved_records:
            display_names = []
            for record_file in saved_records:
                try:
                    record_path = os.path.join(SAVED_RECORDS_DIR, record_file)
                    with open(record_path, 'r', encoding='utf-8') as f:
                        temp_data = json.load(f)
                    display_name = temp_data.get("tipo_ficha", "Desconhecido")
                    patient_name = temp_data.get("paciente", "Sem Paciente")
                    data_preenchimento = datetime.fromisoformat(temp_data.get("data_preenchimento")).strftime("%d/%m/%Y %H:%M")
                    display_names.append(f"{display_name} ({patient_name}) - {data_preenchimento} [{record_file}]")
                except Exception:
                    display_names.append(f"{record_file} (Erro ao ler)")
            
            selected_display_name = st.selectbox("Selecione uma ficha salva para carregar:", 
                                                 [""] + display_names, 
                                                 key="select_saved_record")
            
            selected_record_file = None
            if selected_display_name:
                match = re.search(r'\[(.*?)\]$', selected_display_name)
                if match:
                    selected_record_file = match.group(1)

            if selected_record_file and st.button("Carregar Ficha Salva", key="btn_load_saved_record"):
                record_path = os.path.join(SAVED_RECORDS_DIR, selected_record_file)
                try:
                    with open(record_path, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                    
                    st.session_state.transcricao_geral = loaded_data.get("observacoes_gerais", "")
                    for friendly_name, field_key in FORM_FIELDS_MAP.items():
                        st.session_state[field_key] = loaded_data.get("campos_especificos", {}).get(friendly_name, "")
                    
                    st.session_state.tipo_ficha_aberta = loaded_data.get("tipo_ficha", "Ficha Salva")
                    st.session_state.paciente_atual = loaded_data.get("paciente")
                    
                    model_path = loaded_data.get("modelo_pdf_usado_path")
                    if model_path and os.path.exists(model_path):
                         if model_path not in st.session_state.fichas_pdf_images_cache:
                            st.session_state.fichas_pdf_images_cache[model_path] = get_pdf_images(model_path)
                         st.session_state.current_pdf_images = st.session_state.fichas_pdf_images_cache[model_path]
                    else:
                        st.session_state.current_pdf_images = []

                    st.success(f"Ficha '{selected_record_file}' carregada com sucesso!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Erro ao carregar ficha salva: {e}")
        else:
            st.info("Nenhuma ficha preenchida salva ainda.")


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
