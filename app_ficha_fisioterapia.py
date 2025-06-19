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
def login_page(): # Renomeado para 'login_page'
    st.title("🔐 Login")
    user = st.text_input("Usuário")
    pwd = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fisioterapeuta" and pwd == "1234":
            st.session_state.logado = True
            st.rerun() # Atualiza a página após o login
        else:
            st.error("Usuário ou senha incorretos")

# --- Lógica Principal do Aplicativo ---
# Se não estiver logado, exibe a tela de login.
# Caso contrário, executa o restante do aplicativo.
if not st.session_state.logado:
    login_page() # Chama a página de login
else:
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
                # CORREÇÃO: troquei corrigir_terms por corrigir_termos
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
                        comando_processado = True
                        st.session_state.last_transcription_segment = "" # Limpa o segmento para não adicionar o comando ao campo
                        st.rerun() # Ainda necessário para forçar o destaque
                    else:
                        st.warning(f"Campo '{campo_falado.title()}' não reconhecido.")
                        comando_processado = True # Considera o comando processado, mesmo com erro

                # Comando: "Próximo campo"
                if "proximo campo" in texto_transcrito_lower and not comando_processado:
                    if st.session_state.active_form_field:
                        current_index = FORM_FIELDS_ORDER.index(st.session_state.active_form_field)
                        if current_index < len(FORM_FIELDS_ORDER) - 1:
                            st.session_state.active_form_field = FORM_FIELDS_ORDER[current_index + 1]
                            st.info(f"Campo ativo alterado para: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                        else:
                            st.info("Você está no último campo do formulário.")
                            st.session_state.active_form_field = None # Desativa o campo específico, volta para geral
                    else:
                        if FORM_FIELDS_ORDER:
                            st.session_state.active_form_field = FORM_FIELDS_ORDER[0]
                            st.info(f"Ativando primeiro campo: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                        else:
                            st.warning("Não há campos definidos no formulário.")
                    comando_processado = True
                    st.session_state.last_transcription_segment = "" # Limpa o segmento
                    st.rerun()

                # Comando: "Campo anterior"
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
                            st.session_state.active_form_field = FORM_FIELDS_ORDER[-1] # Vai para o último se não houver ativo
                            st.info(f"Ativando último campo: **{st.session_state.active_form_field.replace('_', ' ').title()}**")
                        else:
                            st.warning("Não há campos definidos no formulário.")
                    comando_processado = True
                    st.session_state.last_transcription_segment = "" # Limpa o segmento
                    st.rerun()

                # Comando: "Parar preenchimento" ou "sair do campo"
                if ("parar preenchimento" in texto_transcrito_lower or
                    "sair do campo" in texto_transcrito_lower) and not comando_processado:
                    if st.session_state.active_form_field:
                        st.info(f"Saindo do campo **{st.session_state.active_form_field.replace('_', ' ').title()}**.")
                        st.session_state.active_form_field = None
                        st.session_state.last_transcription_segment = "" # Limpa o segmento
                        st.rerun()
                        comando_processado = True
                    else:
                        st.warning("Nenhum campo específico está ativo para sair.")
                        comando_processado = True


                # --- Anotação de Texto (se não foi um comando e a anotação está ativa) ---
                if not comando_processado and st.session_state.listening_active:
                    if st.session_state.active_form_field:
                        st.session_state[st.session_state.active_form_field] += " " + texto_transcrito_segmento
                        # Não reruns aqui para anotação contínua nos campos
                    else:
                        st.session_state.transcricao_geral += " " + texto_transcrito_segmento
                        # Não reruns aqui para anotação contínua no geral

                # Limpa o buffer após o processamento
                self.buffer = b""

            return frame # Retorna o frame original (não estamos modificando o áudio em si)

    # --- UI Principal do Aplicativo (visível após o login) ---

    st.title("🗣️ Ficha de Atendimento de Fisioterapia")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Opções de Ficha")

        # Selectbox para Fichas Padrão
        st.subheader("Fichas Padrão")
        for ficha_name in st.session_state.fichas_pdf.keys():
            if st.button(f"Abrir {ficha_name.title()}", key=f"btn_open_pdf_{ficha_name}"):
                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = ficha_name
                st.session_state.transcricao_geral = st.session_state.fichas_pdf[ficha_name]
                for key in FORM_FIELDS_MAP.values(): # Limpa campos específicos
                    st.session_state[key] = ""
                st.session_state.active_form_field = None
                st.success(f"Ficha padrão '{ficha_name.title()}' aberta!")
                st.rerun()

        # Input e Botão para Nova Ficha
        st.subheader("Nova Ficha")
        nova_ficha_tipo = st.text_input("Nome da Nova Ficha (Ex: Avaliação Postural)")
        if st.button("Criar Nova Ficha", key="btn_new_ficha"):
            if nova_ficha_tipo:
                st.session_state.paciente_atual = None
                st.session_state.tipo_ficha_aberta = f"Nova: {nova_ficha_tipo.strip()}"
                st.session_state.transcricao_geral = ""
                for key in FORM_FIELDS_MAP.values(): # Zera todos os campos do formulário
                    st.session_state[key] = ""
                st.session_state.active_form_field = FORM_FIELDS_ORDER[0] if FORM_FIELDS_ORDER else None
                st.success(f"Nova ficha '{nova_ficha_tipo.title()}' criada!")
                st.rerun()
            else:
                st.warning("Por favor, digite o nome da nova ficha.")

        # Seleção de Pacientes Existentes
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
                    for key in FORM_FIELDS_MAP.values(): # Limpa campos específicos
                        st.session_state[key] = ""
                    st.session_state.active_form_field = None
                    st.success(f"Ficha '{ficha_paciente_selecionada.title()}' do paciente '{paciente_selecionado_ui.title()}' aberta!")
                    st.rerun()
                else:
                    st.warning("Por favor, selecione uma ficha para o paciente.")

        st.markdown("---")
        st.header("Controle de Microfone")

        # Botão de Pausar/Retomar anotação
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
        
        # Display do status do microfone (atualizado pela função recv)
        st.markdown(st.session_state.mic_status_message)
        if not st.session_state.listening_active:
            st.warning("Microfone em pausa. Comandos de voz para campos e fichas ainda funcionam.")
        
        st.markdown("---")

    with col2:
        st.header("Conteúdo da Ficha")

        # Exibe o tipo de ficha e o paciente atual
        if st.session_state.tipo_ficha_aberta:
            ficha_titulo = st.session_state.tipo_ficha_aberta.title()
            if st.session_state.paciente_atual:
                st.subheader(f"Ficha: {ficha_titulo} (Paciente: {st.session_state.paciente_atual.title()})")
            else:
                st.subheader(f"Ficha: {ficha_titulo}")
        else:
            st.subheader("Nenhuma ficha aberta")

        # Feedback do último segmento transcrito
        if st.session_state.last_transcription_segment:
            st.markdown(f"<p style='color: grey; font-size: 0.9em;'><i>Última transcrição: \"{st.session_state.last_transcription_segment}\"</i></p>", unsafe_allow_html=True)

        # Se um campo específico está ativo, mostra um destaque
        if st.session_state.active_form_field:
            friendly_name_active = next((k for k, v in FORM_FIELDS_MAP.items() if v == st.session_state.active_form_field), st.session_state.active_form_field)
            st.info(f"Ditando em: **{friendly_name_active.replace('_', ' ').title()}**")


        # Campos específicos do formulário (usando text_area)
        for friendly_name_display, field_key in FORM_FIELDS_MAP.items():
            st.text_area(
                f"{friendly_name_display.title()}:",
                value=st.session_state[field_key],
                key=field_key,
                height=150,
                help=f"Diga 'preencher {friendly_name_display}' para ativar este campo.",
                disabled=(st.session_state.active_form_field != field_key) # Desabilita se não for o campo ativo
            )

        st.markdown("---")
        st.subheader("Observações Gerais da Ficha")
        st.text_area(
            "Texto da Ficha (Geral):",
            value=st.session_state.transcricao_geral,
            key="transcricao_geral_text_area",
            height=300,
            help="Texto ditado sem um campo específico ativo ou carregado de uma ficha existente.",
            disabled=(st.session_state.active_form_field is not None) # Desabilita se houver campo específico ativo
        )

        # Botão para salvar (simulação)
        if st.button("Salvar Ficha (Simulação)", key="btn_save_ficha"):
            st.success("Ficha salva com sucesso! (Esta é uma simulação, os dados não são persistidos).")
            # Aqui você adicionaria a lógica para salvar os dados no seu banco de dados real.
            # Ex: st.session_state.pacientes[algum_paciente][st.session_state.tipo_ficha_aberta] = st.session_state.transcricao_geral

    # --- Configuração e inicialização do WebRTC Streamer ---
    # Colocado aqui para garantir que só inicialize após o login e carregamento do modelo
    webrtc_ctx = webrtc_streamer(
        key="fisioterapia_voice_assistant",
        mode=WebRtcMode.SENDONLY, # Apenas envia áudio, não vídeo
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        async_processing=True,
    )

    # Atualiza o status do microfone baseado na conexão WebRTC
    if webrtc_ctx.state.playing:
        st.session_state.mic_status_message = "🟢 Microfone Conectado (Escutando)"
    else:
        st.session_state.mic_status_message = "🔴 Microfone Desconectado (Aguardando Conexão)"
