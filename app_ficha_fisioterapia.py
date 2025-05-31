import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode # Adicionado WebRtcMode
import av
import whisper
import numpy as np
from datetime import datetime
import os
import re # Adicionado para expressões regulares
import pdfplumber # Adicionado para processamento de PDF

# --- Configurações iniciais ---
st.set_page_config(page_title="Ficha Atendimento - Fisioterapia", layout="centered")

# --- Login simples ---
if "logado" not in st.session_state:
    st.session_state.logado = False

# --- Inicializar armazenamento de fichas em PDF (nome -> texto) ---
if "fichas_pdf" not in st.session_state:
    st.session_state.fichas_pdf = {
        "ficha de anamnese": "Este é o modelo de texto da ficha de anamnese. Inclua histórico médico, queixas principais, cirurgias prévias e medicamentos em uso.",
        "ficha de avaliação ortopédica": "Este é o modelo de texto da ficha de avaliação ortopédica. Detalhe exame físico, testes específicos, postura, marcha e ADMs.",
        # Adicione mais fichas padrão aqui
    }

# --- Inicializar armazenamento de fichas de paciente (simulação de DB) ---
# Na vida real, isso viria de um banco de dados persistente.
if "pacientes" not in st.session_state:
    st.session_state.pacientes = {
        "joao silva": {"anamnese": "Histórico do João: dor no ombro direito há 3 meses, sem cirurgias, usa analgésicos esporadicamente.", "avaliacao_ortopedica": "João Silva, 45 anos, dor ao levantar o braço, ADM reduzida em abdução. Testes de manguito rotador positivos."},
        "maria souza": {"anamnese": "Histórico da Maria: lombalgia crônica, pós-operatório de hérnia de disco há 2 anos, faz uso contínuo de anti-inflamatórios.", "avaliacao_postural": "Maria Souza, 60 anos, assimetria pélvica, hipercifose torácica."},
    }

# --- Novo estado para o comando ativo ---
if "comando_ativo" not in st.session_state:
    st.session_state.comando_ativo = False

# --- Novo estado para armazenar o paciente atual e tipo de ficha aberta ---
if "paciente_atual" not in st.session_state:
    st.session_state.paciente_atual = None
if "tipo_ficha_aberta" not in st.session_state:
    st.session_state.tipo_ficha_aberta = None # Usado para o tipo de ficha do paciente (anamnese, avaliacao_ortopedica)

# --- Funções de Login ---
def login():
    st.title("🔐 Login")
    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fisioterapeuta" and pwd == "1234":
            st.session_state.logado = True
            st.rerun() # Corrigido de experimental_rerun()
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

# --- Inicializar transcrição e outros estados de UI ---
if "transcricao" not in st.session_state:
    st.session_state.transcricao = ""
if "last_transcription_segment" not in st.session_state:
    st.session_state.last_transcription_segment = "" # Para exibir o último segmento

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
            texto_transcrito_segmento = corrigir_termos(result.text).strip()
            st.session_state.last_transcription_segment = texto_transcrito_segmento # Atualiza o último segmento

            comando_processado = False
            texto_transcrito_lower = texto_transcrito_segmento.lower()

            # --- Lógica de Comando (Executada antes da transcrição normal) ---

            # Comando: "abrir ficha de [tipo da ficha padrão]" ou "mostrar [tipo da ficha padrão]"
            match_abrir_ficha_padrao = re.search(r"(?:abrir|mostrar) ficha de (.+)", texto_transcrito_lower)
            if match_abrir_ficha_padrao:
                ficha_solicitada = match_abrir_ficha_padrao.group(1).strip()
                if ficha_solicitada in st.session_state.fichas_pdf:
                    st.session_state.paciente_atual = None # Resetar paciente ao abrir ficha padrão
                    st.session_state.tipo_ficha_aberta = ficha_solicitada # Armazena o tipo de ficha padrão
                    st.session_state.transcricao = st.session_state.fichas_pdf[ficha_solicitada]
                    st.success(f"Ficha padrão '{ficha_solicitada.title()}' aberta e texto carregado!")
                    st.rerun()
                    comando_processado = True
                else:
                    st.warning(f"Comando de ficha padrão '{ficha_solicitada}' não reconhecido.")
                    # Não set comando_processado=True aqui para permitir que a transcrição normal continue,
                    # caso seja apenas uma frase que se assemelha a um comando.

            # Comando: "abrir ficha do paciente [nome do paciente] (de|da)? [tipo da ficha]"
            # Ex: "abrir ficha do paciente joao silva de anamnese"
            match_abrir_paciente_ficha = re.search(r"abrir ficha do paciente (.+?) (?:de|da)? (.+)", texto_transcrito_lower)
            if match_abrir_paciente_ficha and not comando_processado: # Verifica se já não processou um comando
                nome_paciente_falado = match_abrir_paciente_ficha.group(1).strip()
                tipo_ficha_falado = match_abrir_paciente_ficha.group(2).strip()
                
                found_patient = None
                for p_name_db in st.session_state.pacientes:
                    # Busca mais flexível: verifica se o nome falado está contido no nome do DB
                    if nome_paciente_falado in p_name_db: 
                        found_patient = p_name_db
                        break

                if found_patient:
                    if tipo_ficha_falado in st.session_state.pacientes[found_patient]:
                        st.session_state.paciente_atual = found_patient
                        st.session_state.tipo_ficha_aberta = tipo_ficha_falado # Armazena o tipo de ficha do paciente
                        st.session_state.transcricao = st.session_state.pacientes[found_patient][tipo_ficha_falado]
                        st.success(f"Ficha '{tipo_ficha_falado.title()}' do paciente '{found_patient.title()}' aberta e texto carregado!")
                        st.rerun()
                        comando_processado = True
                    else:
                        st.warning(f"Não foi possível encontrar a ficha '{tipo_ficha_falado.title()}' para o paciente '{found_patient.title()}'.")
                        comando_processado = True
                else:
                    st.warning(f"Paciente '{nome_paciente_falado.title()}' não encontrado.")
                    comando_processado = True

            # Comando: "nova ficha de [tipo da ficha]"
            match_nova_ficha = re.search(r"nova ficha de (.+)", texto_transcrito_lower)
            if match_nova_ficha and not comando_processado:
                tipo_nova_ficha = match_nova_ficha.group(1).strip()
                st.session_state.paciente_atual = None # Resetar paciente para nova ficha
                st.session_state.tipo_ficha_aberta = f"Nova: {tipo_nova_ficha}" # Indica que é nova
                st.session_state.transcricao = f"Iniciando nova ficha de {tipo_nova_ficha.title()}. Por favor, dite o conteúdo."
                st.info(f"Preparando para nova ficha: '{tipo_nova_ficha.title()}'. Comece a ditar.")
                st.rerun()
                comando_processado = True

            # --- Transcrição Normal (se nenhum comando foi processado e nenhuma ficha específica está aberta) ---
            # Se não houver ficha aberta ou se for uma "nova ficha" (para ditar o conteúdo)
            if not comando_processado and (st.session_state.tipo_ficha_aberta is None or st.session_state.tipo_ficha_aberta.startswith("Nova:")):
                st.session_state.transcricao += texto_transcrito_segmento + " "
            # Se uma ficha EXISTENTE de paciente/padrão está aberta, e não foi um comando, ditado é adicionado
            elif not comando_processado and st.session_state.tipo_ficha_aberta and not st.session_state.tipo_ficha_aberta.startswith("Nova:"):
                st.session_state.transcricao += texto_transcrito_segmento + " "
            
            self.buffer = b"" # Limpa o buffer
        return frame

# --- Interface ---
st.title("🩺 Ficha de Atendimento - Fisioterapia com IA")

# --- Gerenciamento de Fichas PDF ---
st.subheader("📁 Gerenciamento de Fichas PDF")

with st.expander("Upload e Nomeação de Fichas PDF"):
    st.write("Faça upload de um PDF e dê um nome a ele para acesso futuro. Use nomes claros para os comandos de voz (ex: Ficha de Anamnese, Avaliação Ortopédica).")
    pdf_file = st.file_uploader("Escolha um arquivo PDF", type="pdf", key="pdf_uploader")
    pdf_name_input = st.text_input("Nome para esta Ficha (ex: Ficha de Anamnese)", key="pdf_name_input")

    if st.button("Processar PDF e Salvar Ficha", key="save_pdf_button"):
        if pdf_file is not None and pdf_name_input:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    extracted_text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text: # Evita adicionar None ou strings vazias
                            extracted_text += page_text + "\n"
                
                # Normaliza o nome da ficha para armazenamento e busca
                normalized_pdf_name = pdf_name_input.lower().strip()
                st.session_state.fichas_pdf[normalized_pdf_name] = extracted_text
                st.success(f"✅ Ficha '{pdf_name_input.title()}' salva com sucesso!")
                st.write("--- Conteúdo extraído (amostra) ---")
                st.code(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            except Exception as e:
                st.error(f"❌ Erro ao processar PDF: {e}")
        else:
            st.warning("Por favor, selecione um arquivo PDF e insira um nome para a ficha.")

    st.subheader("Fichas PDF Padrão Disponíveis (Comandos de Voz):")
    if st.session_state.fichas_pdf:
        for name in st.session_state.fichas_pdf.keys():
            st.write(f"- **{name.title()}** (Comando: \"Abrir ficha de {name}\")")
    else:
        st.info("Nenhuma ficha PDF padrão salva ainda. Use a seção acima para adicionar.")

    st.subheader("Fichas de Pacientes (Exemplo - Comandos de Voz):")
    st.write("Ex: \"Abrir ficha do paciente João Silva de anamnese\"")
    if st.session_state.pacientes:
        for paciente, fichas in st.session_state.pacientes.items():
            st.write(f"**{paciente.title()}:**")
            for tipo_ficha in fichas.keys():
                st.write(f"  - {tipo_ficha.title()}")
    else:
        st.info("Nenhuma ficha de paciente de exemplo disponível.")

st.markdown("---") # Separador visual

st.subheader("🎤 Fale e veja o texto ao vivo:")
webrtc_streamer(
    key="microfone",
    mode=WebRtcMode.SENDONLY, # Corrigido aqui
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Exibe o último segmento transcrito para feedback imediato
if st.session_state.last_transcription_segment:
    st.markdown(f"**Última fala:** *{st.session_state.last_transcription_segment}*")

# Exibe qual ficha está ativa e o paciente, se houver
if st.session_state.tipo_ficha_aberta:
    st.info(f"Ficha ativa: **{st.session_state.tipo_ficha_aberta.title()}**")
    if st.session_state.paciente_atual:
        st.info(f"Paciente atual: **{st.session_state.paciente_atual.title()}**")

# Botão para fechar a ficha/limpar a transcrição/resetar o estado
if st.session_state.tipo_ficha_aberta or st.session_state.transcricao: # Mostra o botão se houver algo ativo
    if st.button("Fechar Ficha / Limpar Transcrição"):
        st.session_state.tipo_ficha_aberta = None
        st.session_state.paciente_atual = None
        st.session_state.transcricao = ""
        st.session_state.last_transcription_segment = ""
        st.rerun()

st.text_area(
    "📝 Texto reconhecido / Ficha carregada:",
    st.session_state.transcricao,
    height=300,
    key="main_transcription_area" # Adicionado key para evitar warning
)

st.subheader("📋 Preencha os dados do atendimento")

with st.form("form_ficha"):
    # Se uma ficha de paciente está aberta, preenche o nome e idade (simulação)
    default_nome = ""
    default_idade = 0
    if st.session_state.paciente_atual:
        default_nome = st.session_state.paciente_atual.title()
        # Você precisaria ter a idade no seu 'pacientes' dict
        # Ex: st.session_state.pacientes["joao silva"] = {"idade": 45, ...}
        # Para simplificar, vou deixar a idade como 0 ou você pode adicionar uma idade padrão.

    nome = st.text_input("Nome do paciente", value=default_nome)
    idade = st.number_input("Idade", min_value=0, max_value=120, value=default_idade)
    data = st.date_input("Data do atendimento", value=datetime.today())
    sintomas = st.text_area("Relato do paciente", value=st.session_state.transcricao)
    diagnostico = st.text_area("Diagnóstico clínico")
    conduta = st.text_area("Conduta adotada")
    enviar = st.form_submit_button("Salvar ficha")

    if enviar:
        if not nome:
            st.error("Por favor, preencha o nome do paciente antes de salvar.")
        else:
            # Criar pasta se não existir
            pasta = "fichas_salvas"
            if not os.path.exists(pasta):
                os.makedirs(pasta)
            # Criar nome único para o arquivo
            nome_arquivo = f"{pasta}/ficha_{nome.replace(' ', '_').lower()}_{data.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(nome_arquivo, "w", encoding="utf-8") as f:
                f.write(f"Paciente: {nome}\n")
                f.write(f"Idade: {idade} anos\n")
                f.write(f"Data: {data.strftime('%d/%m/%Y')}\n")
                f.write(f"Relato: {sintomas}\n")
                f.write(f"Diagnóstico: {diagnostico}\n")
                f.write(f"Conduta: {conduta}\n")
            st.success(f"✅ Ficha salva em '{nome_arquivo}'")
            # Limpar transcrição e campos após salvar
            st.session_state.transcricao = ""
            st.session_state.tipo_ficha_aberta = None
            st.session_state.paciente_atual = None
            st.session_state.last_transcription_segment = ""
            st.rerun()
