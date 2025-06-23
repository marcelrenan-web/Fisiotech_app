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
import io # Para lidar com bytes de bytes de imagem
import json # Para salvar metadados de fichas uploadadas

# --- Configurações iniciais da Página Streamlit ---
st.set_page_config(page_title="Ficha Atendimento - Fisioterapia", layout="centered")

# --- Caminhos para Armazenamento ---
UPLOADED_TEMPLATES_DIR = "dados/uploaded_fichas_templates"
UPLOADED_TEMPLATES_INDEX_FILE = "dados/uploaded_fichas_index.json"
PATIENT_RECORDS_FILE = "dados/patient_records.json" # Arquivo para persistir dados de pacientes

# Garante que os diretórios existam
os.makedirs(UPLOADED_TEMPLATES_DIR, exist_ok=True)
# O diretório SAVED_RECORDS_DIR foi removido pois não está sendo usado explicitamente
# mas pode ser recriado se a necessidade de salvar fichas preenchidas individualmente surgir.

# --- Funções para Persistência de Fichas Modelo Uploadadas ---
def load_uploaded_templates_index():
    """Carrega o índice de fichas modelo (templates) uploadadas."""
    if os.path.exists(UPLOADED_TEMPLATES_INDEX_FILE):
        try:
            with open(UPLOADED_TEMPLATES_INDEX_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Erro ao ler {UPLOADED_TEMPLATES_INDEX_FILE}. Criando um novo índice.")
            return {}
    return {}

def save_uploaded_templates_index(index_data):
    """Salva o índice de fichas modelo (templates) uploadadas."""
    with open(UPLOADED_TEMPLATES_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4, ensure_ascii=False)

# --- Funções para Persistência de Dados de Pacientes ---
def load_patient_records():
    """Carrega os registros de pacientes existentes."""
    if os.path.exists(PATIENT_RECORDS_FILE):
        try:
            with open(PATIENT_RECORDS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Erro ao ler {PATIENT_RECORDS_FILE}. Iniciando com dados de pacientes vazios.")
            return {}
    return {}

def save_patient_records(records_data):
    """Salva os registros de pacientes no arquivo JSON."""
    with open(PATIENT_RECORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(records_data, f, indent=4, ensure_ascii=False)

# --- Inicialização de Estados da Sessão Streamlit ---
# Estes estados garantem que o aplicativo mantenha as informações entre as interações do usuário.
if "logado" not in st.session_state:
    st.session_state.logado = False

if "fichas_padrao_paths" not in st.session_state:
    # Defina aqui os caminhos para suas fichas PDF padrão.
    # Certifique-se de que esses arquivos PDF existam no diretório 'dados/'.
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
    st.session_state.pacientes = load_patient_records() # Carrega os registros de pacientes do arquivo

if "tipo_ficha_aberta" not in st.session_state:
    st.session_state.tipo_ficha_aberta = None

if "paciente_atual" not in st.session_state:
    st.session_state.paciente_atual = None

# Alterado para armazenar um dicionário de sessões { "Sessão X": "texto" }
if "conteudo_ficha_atual" not in st.session_state:
    st.session_state.conteudo_ficha_atual = {"Sessão 1": ""} # Inicia com uma sessão padrão

if "sessao_selecionada" not in st.session_state:
    st.session_state.sessao_selecionada = "Sessão 1" # Sessão padrão para edição

if "current_pdf_images" not in st.session_state:
    st.session_state.current_pdf_images = []

if "last_transcription_segment" not in st.session_state:
    st.session_state.last_transcription_segment = ""

if "listening_active" not in st.session_state:
    st.session_state.listening_active = True

if "mic_status_message" not in st.session_state:
    st.session_state.mic_status_message = "🔴 Microfone Desconectado"

# --- Funções Auxiliares de Processamento de PDF ---

@st.cache_data(show_spinner="Extraindo texto do PDF...")
def read_pdf_text(file_path):
    """Extrai texto de um arquivo PDF."""
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
    """Converte as páginas de um PDF em imagens para visualização."""
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

# --- Página de Login ---
def login_page():
    """Exibe a interface de login."""
    st.title("🔐 Login")
    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fisioterapeuta" and pwd == "1234": # Credenciais fixas para demonstração
            st.session_state.logado = True
            st.rerun() # Recarrega a página para mostrar o conteúdo principal
        else:
            st.error("Usuário ou senha incorretos")

# --- Lógica Principal do Aplicativo ---
if not st.session_state.logado:
    login_page() # Exibe a página de login se o usuário não estiver logado
else:
    @st.cache_resource
    def carregar_modelo():
        """Carrega o modelo de transcrição Whisper."""
        st.info("Carregando modelo Whisper (pode levar alguns segundos)...")
        try:
            # Recomenda-se usar um modelo multilíngue maior ou um específico para pt-BR.
            # "base" é multilíngue, "small" é multilíngue e mais preciso.
            model = whisper.load_model("base") # Ou "small", "medium" para melhor precisão
            st.success("Modelo Whisper 'base' carregado!")
        except Exception as e:
            st.error(f"Erro ao carregar modelo Whisper: {e}. Verifique sua conexão ou instalação.")
            st.stop() # Interrompe a execução se o modelo não carregar
        return model

    model = carregar_modelo()

    def corrigir_termos(texto):
        """Aplica correções a termos comuns na transcrição (ajustável)."""
        correcoes = {
            "tendinite": "tendinite",
            "cervicalgia": "cervicalgia",
            "lombar": "região lombar",
            "reabilitação funcional": "reabilitação funcional",
            "fisioterapia do ombro": "fisioterapia de ombro",
            "dor nas costas": "algia na coluna",
            # Adicione mais correções específicas da área de fisioterapia conforme necessário
        }
        for errado, certo in correcoes.items():
            texto = texto.replace(errado, certo)
        return texto

    class AudioProcessor(AudioProcessorBase):
        """Processador de áudio para transcrição em tempo real e comandos de voz."""
        def __init__(self) -> None:
            self.buffer = b""

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            """Recebe quadros de áudio, processa e executa comandos/transcrições."""
            pcm = frame.to_ndarray().flatten().astype(np.float32).tobytes()
            self.buffer += pcm

            # Processa a cada 5 segundos de áudio acumulado
            if len(self.buffer) > 32000 * 5:
                audio_np = np.frombuffer(self.buffer, np.float32)
                audio_np = whisper.pad_or_trim(audio_np)
                
                # Garante que o modelo esteja no dispositivo correto (CPU ou CUDA)
                mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
                options = whisper.DecodingOptions(language="pt", fp16=False) # Especifica o idioma português
                result = whisper.decode(model, mel, options)
                
                texto_transcrito_segmento = corrigir_termos(result.text).strip()
                st.session_state.last_transcription_segment = texto_transcrito_segmento
                
                comando_processado = False
                texto_transcrito_lower = texto_transcrito_segmento.lower()

                # --- Comandos de controle de escuta (pausar/retomar anotação) ---
                if "pausar anotação" in texto_transcrito_lower:
                    st.session_state.listening_active = False
                    st.session_state.last_transcription_segment = "" # Limpa a exibição do comando
                    comando_processado = True
                elif "retomar anotação" in texto_transcrito_lower:
                    st.session_state.listening_active = True
                    st.session_state.last_transcription_segment = "" # Limpa a exibição do comando
                    comando_processado = True

                # --- Comandos para navegar entre sessões ---
                match_mudar_sessao = re.search(r"ir para a sessão (\d+)", texto_transcrito_lower)
                if match_mudar_sessao and not comando_processado:
                    sessao_num = int(match_mudar_sessao.group(1))
                    nova_sessao_nome = f"Sessão {sessao_num}"
                    if nova_sessao_nome in st.session_state.conteudo_ficha_atual:
                        st.session_state.sessao_selecionada = nova_sessao_nome
                        st.success(f"Mudou para a {nova_sessao_nome}.")
                        comando_processado = True
                        st.rerun()
                    else:
                        st.warning(f"Sessão '{nova_sessao_nome}' não existe. Crie-a primeiro.")
                        comando_processado = True
                
                if "nova sessão" in texto_transcrito_lower and not comando_processado:
                    proxima_sessao_num = len(st.session_state.conteudo_ficha_atual) + 1
                    nova_sessao_nome = f"Sessão {proxima_sessao_num}"
                    st.session_state.conteudo_ficha_atual[nova_sessao_nome] = ""
                    st.session_state.sessao_selecionada = nova_sessao_nome
                    st.success(f"Nova {nova_sessao_nome} criada.")
                    comando_processado = True
                    st.rerun()

                # --- Lógica para abrir Ficha Modelo (PDF padrão ou uploadado) via comando de voz ---
                match_abrir_ficha_modelo = re.search(r"(?:abrir|mostrar) ficha de (.+)", texto_transcrito_lower)
                if match_abrir_ficha_modelo and not comando_processado:
                    ficha_solicitada = match_abrir_ficha_modelo.group(1).strip()
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

                        st.session_state.paciente_atual = None # Nenhuma paciente associado ao abrir um modelo
                        st.session_state.tipo_ficha_aberta = ficha_solicitada
                        st.session_state.conteudo_ficha_atual = {"Sessão 1": ""} # Inicia nova ficha com uma sessão padrão
                        st.session_state.sessao_selecionada = "Sessão 1"
                        st.success(f"Ficha '{ficha_solicitada.title()}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                        comando_processado = True
                        st.rerun() # Força um rerun para atualizar a UI imediatamente com a nova ficha
                    else:
                        st.warning(f"Comando de ficha modelo '{ficha_solicitada}' não reconhecido.")
                        comando_processado = True # Marca como processado para não adicionar ao texto geral

                # --- Lógica para abrir ficha de paciente existente via comando de voz ---
                match_abrir_paciente_ficha = re.search(r"abrir ficha do paciente (.+?) (?:de|da)? (.+)", texto_transcrito_lower)
                if match_abrir_paciente_ficha and not comando_processado:
                    nome_paciente_falado = match_abrir_paciente_ficha.group(1).strip()
                    tipo_ficha_falado = match_abrir_paciente_ficha.group(2).strip()
                    
                    found_patient = None
                    # Busca parcial pelo nome do paciente para maior flexibilidade
                    for p_name_db in st.session_state.pacientes:
                        if nome_paciente_falado in p_name_db:
                            found_patient = p_name_db
                            break

                    if found_patient:
                        if tipo_ficha_falado in st.session_state.pacientes[found_patient]:
                            st.session_state.paciente_atual = found_patient
                            st.session_state.tipo_ficha_aberta = tipo_ficha_falado
                            # Carrega o dicionário de sessões da ficha do paciente
                            st.session_state.conteudo_ficha_atual = st.session_state.pacientes[found_patient][tipo_ficha_falado]
                            
                            # Define a sessão selecionada para a primeira existente ou uma padrão
                            if st.session_state.conteudo_ficha_atual:
                                st.session_state.sessao_selecionada = list(st.session_state.conteudo_ficha_atual.keys())[0]
                            else:
                                st.session_state.conteudo_ficha_atual = {"Sessão 1": ""}
                                st.session_state.sessao_selecionada = "Sessão 1"

                            st.session_state.current_pdf_images = [] # Limpa visualização de PDF
                            
                            st.success(f"Ficha '{tipo_ficha_falado.title()}' do paciente '{found_patient.title()}' aberta e texto carregado!")
                            comando_processado = True
                            st.rerun() # Força um rerun para atualizar a UI
                        else:
                            st.warning(f"Não foi possível encontrar a ficha '{tipo_ficha_falado.title()}' para o paciente '{found_patient.title()}'.")
                            comando_processado = True
                    else:
                        st.warning(f"Paciente '{nome_paciente_falado.title()}' não encontrado.")
                        comando_processado = True

                # --- Lógica para criar uma nova ficha em branco via comando de voz ---
                match_nova_ficha = re.search(r"nova ficha de (.+)", texto_transcrito_lower)
                if match_nova_ficha and not comando_processado:
                    tipo_nova_ficha = match_nova_ficha.group(1).strip()
                    st.session_state.paciente_atual = None # Não há paciente associado inicialmente
                    st.session_state.tipo_ficha_aberta = f"Nova: {tipo_nova_ficha}" # Prefixo para indicar nova ficha
                    st.session_state.conteudo_ficha_atual = {"Sessão 1": ""} # Nova ficha com uma sessão padrão
                    st.session_state.sessao_selecionada = "Sessão 1"
                    
                    st.session_state.current_pdf_images = [] # Limpa visualização de PDF
                    
                    st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Dite na Sessão 1.")
                    comando_processado = True
                    st.rerun() # Força um rerun para atualizar a UI

                # --- Adiciona a transcrição ao campo da sessão atual se nenhum comando foi processado e a escuta está ativa ---
                if not comando_processado and st.session_state.listening_active:
                    if texto_transcrito_segmento and st.session_state.sessao_selecionada:
                        # Adiciona ao conteúdo da sessão atualmente selecionada
                        current_text_for_session = st.session_state.conteudo_ficha_atual.get(st.session_state.sessao_selecionada, "")
                        st.session_state.conteudo_ficha_atual[st.session_state.sessao_selecionada] = current_text_for_session + " " + texto_transcrito_segmento
                
                self.buffer = b"" # Limpa o buffer de áudio após o processamento

            return frame

    # --- Layout da Interface do Usuário (UI) Principal ---

    st.title("🗣️ Ficha de Atendimento de Fisioterapia")
    st.markdown("---")

    col1, col2 = st.columns([1, 2]) # Divide a tela em duas colunas

    with col1: # Coluna da esquerda para opções e controles
        st.header("Opções de Ficha")

        # --- Seção de Upload de Fichas Modelo ---
        st.subheader("Upload de Nova Ficha Modelo (PDF)")
        uploaded_file = st.file_uploader("Escolha um arquivo PDF para upload", type="pdf", key="file_uploader_template")
        new_ficha_name = st.text_input("Nome para esta nova ficha modelo (Ex: 'Ficha de Coluna')", key="new_uploaded_ficha_name")
        
        if uploaded_file is not None and new_ficha_name:
            if st.button("Salvar Ficha Modelo Uploaded", key="btn_save_uploaded_template"):
                # Sanitiza o nome para uso no sistema de arquivos
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
                    st.session_state.new_uploaded_ficha_name = "" # Limpa o campo de nome
                    st.rerun() # Recarrega para limpar o uploader e o text_input
                except Exception as e:
                    st.error(f"Erro ao salvar o arquivo: {e}")
        elif uploaded_file is None and new_ficha_name:
            st.warning("Por favor, selecione um arquivo PDF antes de salvar a ficha modelo.")

        st.markdown("---")

        # --- Seleção de Fichas Modelo (Padrão e Uploadadas) ---
        st.subheader("Abrir Ficha Modelo (PDF)")
        
        all_template_fichas = {}
        # Adiciona fichas padrão à lista de opções
        for name, path in st.session_state.fichas_padrao_paths.items():
            if os.path.exists(path):
                all_template_fichas[name.lower()] = {"name": name, "path": path}
            else:
                st.warning(f"Ficha Padrão '{name}' não encontrada em '{path}'. Verifique o diretório 'dados/'.")
        
        # Adiciona fichas uploadadas à lista e remove aquelas que não existem mais no disco
        keys_to_remove = []
        for key, info in st.session_state.uploaded_fichas_data.items():
            if os.path.exists(info['path']):
                all_template_fichas[key] = info
            else:
                keys_to_remove.append(key)
        
        if keys_to_remove:
            for key in keys_to_remove:
                st.warning(f"Ficha Modelo '{st.session_state.uploaded_fichas_data[key]['name']}' não encontrada em '{st.session_state.uploaded_fichas_data[key]['path']}'. Será removida da lista.")
                del st.session_state.uploaded_fichas_data[key]
            save_uploaded_templates_index(st.session_state.uploaded_fichas_data)
            st.rerun() # Recarrega a lista após remoção

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
                st.session_state.conteudo_ficha_atual = {"Sessão 1": ""} # Inicia nova ficha com uma sessão padrão
                st.session_state.sessao_selecionada = "Sessão 1"
                st.success(f"Ficha modelo '{selected_template_ficha_name}' aberta. Veja o PDF como guia e insira as respostas abaixo.")
                st.rerun()
            else:
                st.error("Erro ao encontrar o caminho da ficha selecionada.")

        # --- Opção para deletar fichas modelo salvas ---
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
                        
                        # Limpa o cache se a ficha deletada estava em memória
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
                st.session_state.conteudo_ficha_atual = {"Sessão 1": ""} # Nova ficha com uma sessão padrão
                st.session_state.sessao_selecionada = "Sessão 1"
                st.session_state.current_pdf_images = []
                st.success(f"Nova ficha '{nova_ficha_tipo.title()}' criada! Dite na Sessão 1.")
                st.rerun()
            else:
                st.warning("Por favor, digite o nome da nova ficha.")

        st.subheader("Fichas de Pacientes Existentes")
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
                # Carrega o dicionário de sessões da ficha do paciente
                st.session_state.conteudo_ficha_atual = st.session_state.pacientes[paciente_selecionado_ui][ficha_paciente_selecionada]
                
                # Define a sessão selecionada para a primeira existente ou uma padrão
                if st.session_state.conteudo_ficha_atual:
                    st.session_state.sessao_selecionada = list(st.session_state.conteudo_ficha_atual.keys())[0]
                else: # Caso a ficha esteja vazia por algum motivo
                    st.session_state.conteudo_ficha_atual = {"Sessão 1": ""}
                    st.session_state.sessao_selecionada = "Sessão 1"

                st.session_state.current_pdf_images = [] # Limpa a visualização do PDF de modelo
                st.success(f"Ficha '{ficha_paciente_selecionada.title()}' do paciente '{paciente_selecionado_ui.title()}' aberta e texto carregado!")
                st.rerun()
            else:
                st.warning("Por favor, selecione um paciente e uma ficha para abrir.")

        st.markdown("---")
        st.header("Controle de Microfone")

        # Componente Streamlit WebRTC para captura de áudio
        webrtc_ctx = webrtc_streamer(
            key="audio_recorder_streamer",
            mode=WebRtcMode.SENDONLY, # Apenas envia áudio
            audio_processor_factory=AudioProcessor, # Usa nossa classe para processar o áudio
            rtc_configuration=RTCConfiguration( # Configuração para STUN/TURN servers
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": False, "audio": True}, # Apenas áudio
            async_processing=True, # Permite processamento assíncrono
        )
        
        # Atualiza o status do microfone na UI
        if webrtc_ctx.state.playing:
            st.session_state.mic_status_message = "🟢 Microfone Conectado (Ouvindo)"
        else:
            st.session_state.mic_status_message = "🔴 Microfone Desconectado"

        # Botões para pausar/retomar a anotação de voz
        if st.session_state.listening_active:
            if st.button("Pausar Anotação de Voz ⏸️", key="btn_pause_listening"):
                st.session_state.listening_active = False
                st.info("Anotação de voz pausada. Diga 'retomar anotação' para continuar.")
        else:
            if st.button("Retomar Anotação de Voz ▶️", key="btn_resume_listening"):
                st.session_state.listening_active = True
                st.info("Anotação de voz retomada.")
                
        st.markdown(st.session_state.mic_status_message)
        if not st.session_state.listening_active:
            st.warning("Microfone em pausa. Comandos de voz para abrir fichas ainda funcionam, mas o ditado geral está pausado.")
            
        st.markdown("---")

    with col2: # Coluna da direita para o conteúdo da ficha e transcrição
        st.header("Conteúdo da Ficha")

        # Exibe o título da ficha atualmente aberta
        if st.session_state.tipo_ficha_aberta:
            ficha_titulo = st.session_state.tipo_ficha_aberta.title()
            if st.session_state.paciente_atual:
                st.subheader(f"Ficha: {ficha_titulo} (Paciente: {st.session_state.paciente_atual.title()})")
            else:
                # Remove o prefixo "Nova:" para exibição amigável
                display_title = ficha_titulo.replace("Nova: ", "") if "nova:" in ficha_titulo.lower() else ficha_titulo
                st.subheader(f"Ficha: {display_title}")
        else:
            st.subheader("Nenhuma ficha aberta")

        # Exibe o último segmento de transcrição para feedback em tempo real
        if st.session_state.last_transcription_segment:
            st.markdown(f"<p style='color: grey; font-size: 0.9em;'><i>Última transcrição: \"{st.session_state.last_transcription_segment}\"</i></p>", unsafe_allow_html=True)

        # Exibe as imagens do PDF da ficha modelo, se houver uma aberta
        if st.session_state.current_pdf_images:
            st.subheader("Visualização da Ficha (Guia PDF)")
            for i, img in enumerate(st.session_state.current_pdf_images):
                st.image(img, caption=f"Página {i+1} do PDF", use_column_width=True)
            st.markdown("---")

        # --- Seção de Navegação e Edição de Sessões ---
        if st.session_state.tipo_ficha_aberta:
            st.subheader("Sessões da Ficha")

            # Permite adicionar uma nova sessão
            if st.button("Adicionar Nova Sessão", key="btn_add_session"):
                next_session_num = len(st.session_state.conteudo_ficha_atual) + 1
                new_session_name = f"Sessão {next_session_num}"
                st.session_state.conteudo_ficha_atual[new_session_name] = ""
                st.session_state.sessao_selecionada = new_session_name # Seleciona a nova sessão automaticamente
                st.info(f"Sessão '{new_session_name}' adicionada.")
                st.rerun()

            # Selector para escolher a sessão atual
            session_options = list(st.session_state.conteudo_ficha_atual.keys())
            st.session_state.sessao_selecionada = st.selectbox(
                "Selecione a Sessão para editar:",
                options=session_options,
                index=session_options.index(st.session_state.sessao_selecionada) if st.session_state.sessao_selecionada in session_options else 0,
                key="session_selector"
            )

            # Campo de texto para a sessão selecionada
            current_session_text = st.session_state.conteudo_ficha_atual.get(st.session_state.sessao_selecionada, "")
            
            # Campo principal para as observações da sessão atual, onde a transcrição é inserida
            st.session_state.conteudo_ficha_atual[st.session_state.sessao_selecionada] = st.text_area(
                f"Texto da {st.session_state.sessao_selecionada}:",
                value=current_session_text,
                key=f"transcricao_sessao_{st.session_state.sessao_selecionada}", # Chave única por sessão
                height=300,
                help="Todo o ditado por voz, a menos que seja um comando, será inserido aqui."
            )
        else:
            st.info("Abra ou crie uma ficha para começar a ditar e gerenciar sessões.")


        st.markdown("---")
        # Botão para salvar a ficha
        if st.session_state.tipo_ficha_aberta:
            if st.button("Salvar Ficha", key="btn_save_ficha"):
                if st.session_state.paciente_atual:
                    # Se há um paciente atual, atualiza a ficha existente
                    st.session_state.pacientes[st.session_state.paciente_atual][st.session_state.tipo_ficha_aberta] = st.session_state.conteudo_ficha_atual
                    save_patient_records(st.session_state.pacientes) # Salva as alterações no arquivo
                    st.success(f"Ficha de '{st.session_state.tipo_ficha_aberta.title()}' do paciente '{st.session_state.paciente_atual.title()}' atualizada com sucesso!")
                    st.rerun() # Recarrega para limpar a mensagem de "Nome do Paciente para Salvar" se ela apareceu
                elif st.session_state.tipo_ficha_aberta.startswith("Nova:"):
                    # Se é uma nova ficha em branco, pede o nome do paciente para salvar
                    new_patient_name_for_save = st.text_input("Nome do Paciente para Salvar a Nova Ficha:", key="new_patient_name_save_on_save_button")
                    if new_patient_name_for_save:
                        patient_key = new_patient_name_for_save.lower().strip()
                        if patient_key not in st.session_state.pacientes:
                            st.session_state.pacientes[patient_key] = {} # Cria uma nova entrada para o paciente se não existir
                        
                        ficha_name_to_save = st.session_state.tipo_ficha_aberta.replace("Nova: ", "").strip().lower()
                        st.session_state.pacientes[patient_key][ficha_name_to_save] = st.session_state.conteudo_ficha_atual
                        save_patient_records(st.session_state.pacientes) # Salva no arquivo
                        st.success(f"Nova ficha '{ficha_name_to_save.title()}' salva para o paciente '{patient_key.title()}'!")
                        st.session_state.paciente_atual = patient_key # Define o paciente e ficha como os atuais
                        st.session_state.tipo_ficha_aberta = ficha_name_to_save
                        st.rerun() # Recarrega para atualizar a UI e remover o campo de input
                    else:
                        st.warning("Por favor, digite o nome do paciente para salvar a nova ficha.")
                else:
                    st.warning("Não é possível salvar a ficha. Abra uma ficha existente de paciente ou crie uma nova ficha em branco.")
        else:
            st.info("Abra ou crie uma ficha para começar a ditar.")
