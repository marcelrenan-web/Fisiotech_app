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

# --- Classe para Processamento de Áudio (Core da Transcrição e Comandos de Voz) ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffer = b"" # Buffer para acumular áudio

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.float32).tobytes()
        self.buffer += pcm

        # Processa o buffer a cada 5 segundos de áudio acumulado
        if len(self.buffer) > 32000 * 5:  # 32000 samples/seg * 5 segundos
            audio_np = np.frombuffer(self.buffer, np.float32)
            audio_np = whisper.pad_or_trim(audio_np) # Ajusta o tamanho do áudio para o Whisper
            mel = whisper.log_mel_spectrogram(audio_np).to(model.device) # Converte para mel spectrogram
            options = whisper.DecodingOptions(language="pt", fp16=False) # Opções de decodificação (português)
            result = whisper.decode(model, mel, options) # Decodifica o áudio para texto
            
            # Corrige e normaliza o texto transcrito
            texto_transcrito_segmento = corrigir_termos(result.text).strip()
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
                    st.rerun() # Ainda necessário para forçar o destaque visual do campo
                else:
                    st.warning(f"Campo '{campo_falado.title()}' não reconhecido. Tente novamente.")
                    comando_processado = True # Considera o comando processado (mas com erro)

            # Comando: "Próximo campo"
            if "próximo campo" in texto_transcrito_lower and not comando_processado:
                if st.session_state.active_form_field:
                    try:
                        current_index = FORM_FIELDS_ORDER.index(st.session_state.active_form_field)
                        next_index = (current_index + 1) % len(FORM_FIELDS_ORDER) # Navega para o próximo, ou volta ao início
                        st.session_state.active_form_field = FORM_FIELDS_ORDER[next_index]
                        comando_processado = True
                        st.session_state.last_transcription_segment = ""
                        st.rerun()
                    except ValueError: # Caso o campo ativo não esteja na lista de ordem (improvável)
                        st.warning("Não foi possível encontrar o próximo campo na sequência.")
                        comando_processado = True
                else:
                    # Se não há campo ativo, ativa o primeiro da lista
                    if FORM_FIELDS_ORDER:
                        st.session_state.active_form_field = FORM_FIELDS_ORDER[0]
                        comando_processado = True
                        st.session_state.last_transcription_segment = ""
                        st.rerun()
                    else:
                        st.warning("Não há campos definidos para navegação sequencial.")
                        comando_processado = True

            # Comando: "Finalizar campo" ou "Parar preenchimento"
            if ("finalizar campo" in texto_transcrito_lower or "parar preenchimento" in texto_transcrito_lower) and not comando_processado:
                if st.session_state.active_form_field:
                    st.success(f"✅ Finalizado ditado para: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                    st.session_state.active_form_field = None # Desativa o campo
                    # st.info("🎤 Voltando para ditado geral. Para preencher um campo específico, diga 'Preencher [Nome do Campo]'.")
                    comando_processado = True
                    st.session_state.last_transcription_segment = ""
                    st.rerun()
                else:
                    st.warning("Nenhum campo ativo para finalizar.")
                    comando_processado = True

            # --- Lógica de Transcrição Normal (Ditado de Conteúdo) ---
            # O texto transcrito é adicionado ou ao campo ativo ou à transcrição geral
            # APENAS SE A ANOTAÇÃO ESTIVER ATIVA e NÃO FOR UM COMANDO
            if not comando_processado and texto_transcrito_segmento and st.session_state.listening_active:
                if st.session_state.active_form_field:
                    # Adiciona o texto ao campo de formulário atualmente ativo
                    st.session_state[st.session_state.active_form_field] += texto_transcrito_segmento + " "
                else:
                    # Adiciona o texto à área de transcrição geral
                    st.session_state.transcricao_geral += texto_transcrito_segmento + " "
            
            self.buffer = b"" # Limpa o buffer de áudio para o próximo segmento
        return frame

# --- Estrutura da Interface do Streamlit (UI) ---
st.title("🩺 Ficha de Atendimento - Fisioterapia com IA")

# --- Seção: Gerenciamento de Fichas PDF Modelo ---
st.subheader("📁 Gerenciamento de Fichas PDF Modelo")

with st.expander("Upload e Nomeação de Novas Fichas PDF"):
    st.write("Faça upload de um PDF com um modelo de ficha (somente texto simples) e dê um nome amigável a ele para acesso futuro via comando de voz. Ex: 'Ficha de Anamnese', 'Avaliação Ortopédica'.")
    pdf_file = st.file_uploader("Escolha um arquivo PDF", type="pdf", key="pdf_uploader")
    pdf_name_input = st.text_input("Nome para esta Ficha (ex: Ficha de Anamnese)", key="pdf_name_input")

    if st.button("Processar PDF e Salvar Modelo de Ficha", key="save_pdf_button"):
        if pdf_file is not None and pdf_name_input:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    extracted_text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text: # Garante que só adiciona texto existente
                            extracted_text += page_text + "\n"
                
                # Normaliza o nome da ficha para armazenamento e busca (minúsculas e sem espaços extras)
                normalized_pdf_name = pdf_name_input.lower().strip()
                st.session_state.fichas_pdf[normalized_pdf_name] = extracted_text
                st.success(f"✅ Modelo de ficha '{pdf_name_input.title()}' salvo com sucesso!")
                st.write("--- Conteúdo extraído (amostra) ---")
                st.code(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
            except Exception as e:
                st.error(f"❌ Erro ao processar PDF: {e}. Certifique-se de que é um PDF de texto e não uma imagem.")
        else:
            st.warning("Por favor, selecione um arquivo PDF e insira um nome para o modelo da ficha.")

    st.subheader("Modelos de Fichas Padrão Disponíveis (Comandos de Voz):")
    st.info("Para abrir uma ficha modelo, diga: \"Abrir ficha de [Nome da Ficha]\"")
    if st.session_state.fichas_pdf:
        for name in st.session_state.fichas_pdf.keys():
            st.write(f"- **{name.title()}**")
    else:
        st.info("Nenhum modelo de ficha PDF padrão salvo ainda. Use a seção acima para adicionar.")

    st.subheader("Fichas de Pacientes de Exemplo (Comandos de Voz):")
    st.info("Para abrir a ficha de um paciente, diga: \"Abrir ficha do paciente [Nome do Paciente] [Tipo da Ficha]\"")
    if st.session_state.pacientes:
        for paciente, fichas in st.session_state.pacientes.items():
            st.write(f"**{paciente.title()}:**")
            for tipo_ficha in fichas.keys():
                st.write(f"  - {tipo_ficha.title()}")
    else:
        st.info("Nenhuma ficha de paciente de exemplo disponível.")

st.markdown("---") # Separador visual

# --- Seção de Controle de Microfone e Transcrição ---
st.subheader("🎤 Controle de Microfone e Transcrição")

# NOVO: Configuração para melhorar a conexão WebRTC (opcional, mas recomendado)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="microfone",
    mode=WebRtcMode.SENDONLY, # Modo SENDONLY: Envia áudio do microfone para o processador
    audio_processor_factory=AudioProcessor, # Nossa classe customizada para processar o áudio
    media_stream_constraints={"audio": True, "video": False}, # Captura apenas áudio
    rtc_configuration=RTC_CONFIGURATION, # Aplica a configuração STUN
    # NOVO: Callback para atualizar o estado da conexão do microfone
    on_connection_state_changed=lambda state: st.session_state.update(
        mic_status_message="🟢 Microfone Conectado e Escutando" if state.is_connected else "🔴 Microfone Desconectado"
    )
)

# NOVO: Exibe o status da conexão do microfone
st.markdown(f"**Status da Conexão do Microfone:** {st.session_state.mic_status_message}")

# Exibe o status da anotação (pausada ou ativa)
if not st.session_state.listening_active:
    st.warning("⏸️ Anotação Pausada. Diga 'Retomar anotação' para continuar a ditar.")
else:
    # Exibe o último segmento transcrito para feedback imediato ao usuário
    if st.session_state.last_transcription_segment:
        st.markdown(f"**Última fala:** *{st.session_state.last_transcription_segment}*")


# Exibe qual ficha (modelo ou de paciente) está ativa e o paciente atual
if st.session_state.tipo_ficha_aberta:
    st.info(f"Ficha ativa: **{st.session_state.tipo_ficha_aberta.title()}**")
    if st.session_state.paciente_atual:
        st.info(f"Paciente atual: **{st.session_state.paciente_atual.title()}**")

# Exibe qual campo do formulário está ativo para ditado por voz
if st.session_state.active_form_field:
    st.markdown(f"**Ditando para o campo:** <span style='background-color: #ffeb3b; padding: 5px; border-radius: 5px;'>{st.session_state.active_form_field.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
else:
    st.info("Nenhum campo específico ativo. O ditado irá para 'Observações Gerais'. Diga 'Preencher [Nome do Campo]' para ativar um campo.")
    st.markdown("**Comandos de navegação:**")
    st.markdown("- **\"Preencher [Nome do Campo]\"** (ex: \"Preencher sono\")")
    st.markdown("- **\"Próximo campo\"**")
    st.markdown("- **\"Finalizar campo\"** ou **\"Parar preenchimento\"**")
    st.markdown("- **\"Pausar anotação\"**")
    st.markdown("- **\"Retomar anotação\"**")


# Botão para fechar a ficha atual / limpar toda a transcrição e estados
if st.session_state.tipo_ficha_aberta or st.session_state.transcricao_geral or any(st.session_state[key] for key in FORM_FIELDS_MAP.values()):
    if st.button("🔴 Fechar Ficha / Limpar Tudo"):
        st.session_state.tipo_ficha_aberta = None
        st.session_state.paciente_atual = None
        st.session_state.transcricao_geral = ""
        st.session_state.active_form_field = None
        st.session_state.last_transcription_segment = ""
        st.session_state.listening_active = True # Reseta para ativo ao limpar
        st.session_state.mic_status_message = "🔴 Microfone Desconectado" # Reseta o status
        # Limpa todos os campos do formulário
        for key in FORM_FIELDS_MAP.values():
            st.session_state[key] = ""
        st.rerun()

st.markdown("---") # Separador visual

# --- Formulário de Preenchimento da Ficha ---
st.subheader("📋 Preencha os dados do atendimento")

with st.form("form_ficha"):
    # Preenchimento inicial do nome do paciente e idade (se um paciente estiver ativo)
    default_nome = ""
    default_idade = 0 # Idade precisa ser adicionada ao dicionário de pacientes se for preenchida automaticamente
    if st.session_state.paciente_atual:
        default_nome = st.session_state.paciente_atual.title()
        # Ex: se st.session_state.pacientes["joao silva"] = {"idade": 45, ...}
        # default_idade = st.session_state.pacientes[st.session_state.paciente_atual].get("idade", 0)

    nome = st.text_input("Nome do paciente", value=default_nome)
    idade = st.number_input("Idade", min_value=0, max_value=120, value=default_idade)
    data = st.date_input("Data do atendimento", value=datetime.today())

    st.markdown("---") # Separador visual

    st.subheader("Detalhes da Anamnese/Avaliação:")

    # Campos individuais do formulário para cada tópico
    for friendly_name, field_key in FORM_FIELDS_MAP.items():
        st.text_area(
            f"**{friendly_name.replace('_', ' ').title()}**", # Título formatado para exibição
            value=st.session_state[field_key], # Vincula ao estado da sessão correspondente
            key=f"form_field_{field_key}", # Chave única para o widget
            height=150, # Altura do campo
        )

    # Campo para ditado geral/observações (se não estiver ditando em campo específico)
    st.text_area(
        "**Observações Gerais (Ditado livre ou carregado de ficha modelo):**",
        value=st.session_state.transcricao_geral,
        height=200,
        key="observacoes_gerais_input"
    )
    
    st.markdown("---") # Separador visual

    diagnostico = st.text_area("Diagnóstico clínico")
    conduta = st.text_area("Conduta adotada")
    enviar = st.form_submit_button("Salvar ficha")

    if enviar:
        if not nome:
            st.error("Por favor, preencha o nome do paciente antes de salvar.")
        else:
            # Garante que a pasta de salvamento exista
            pasta = "fichas_salvas"
            if not os.path.exists(pasta):
                os.makedirs(pasta)
            
            # Cria um nome de arquivo único para a ficha
            nome_arquivo = f"{pasta}/ficha_{nome.replace(' ', '_').lower()}_{data.strftime('%Y%m%d_%H%M%S')}.txt"
            
            # Salva os dados no arquivo de texto
            with open(nome_arquivo, "w", encoding="utf-8") # <<< CORREÇÃO AQUI
