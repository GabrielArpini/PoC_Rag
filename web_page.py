import streamlit as st
from pre_processamento import processar_item_unico
from query_processing import processar_query
import os 

# Page configuration
st.set_page_config(
    page_title="Assistente RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'processando' not in st.session_state:
    st.session_state.processando = False
if 'arquivos_processados' not in st.session_state:
    st.session_state.arquivos_processados = []

# Header
st.title("Assistente RAG")
st.caption("Faça perguntas sobre seus documentos ou busque informações na web")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Status")
    st.metric("Arquivos", len(st.session_state.arquivos_processados))
    
    st.divider()
    
    if st.session_state.arquivos_processados:
        st.subheader("Documentos")
        for filename in st.session_state.arquivos_processados:
            icon = "📄" if filename.endswith('.txt') else "📕"
            st.text(f"{icon} {filename}")
    else:
        st.info("Nenhum arquivo ainda")

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload")
    uploaded_file = st.file_uploader(
        "Selecione arquivos",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )

with col2:
    st.subheader("Pergunta")
    query = st.text_input(
        "Digite sua pergunta:",
        placeholder="Ex: O que é radiação cósmica de fundo?"
    )
    
    processar_btn = st.button(
        "Processar",
        disabled=not query,
        type="primary"
    )

# File processing
if uploaded_file:
    st.divider()
    st.subheader("Processamento")
    
    progress_bar = st.progress(0)
    total = len(uploaded_file)
    
    for idx, file in enumerate(uploaded_file):
        if file.name in st.session_state.arquivos_processados:
            continue
        
        extensao = file.name.split('.')[-1]
        save_dir = 'data/pdfs' if extensao == 'pdf' else 'data/txts'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file.name)
        
        with st.status(f"Processando {file.name}...", expanded=True) as status:
            try:
                with open(save_path, 'wb') as f:
                    f.write(file.getbuffer())
                
                num_chunks = processar_item_unico(save_path, extensao)
                st.session_state.arquivos_processados.append(file.name)
                
                status.update(label=f"✅ {file.name} concluído!", state="complete")
                st.success(f"{num_chunks} chunks processados")
                
            except Exception as e:
                status.update(label=f"❌ Erro em {file.name}", state="error")
                st.error(str(e))
        
        progress_bar.progress((idx + 1) / total)

# Query processing
if processar_btn and query:
    st.divider()
    
    with st.spinner("Buscando resposta..."):
        resposta, contextos, usou_web,usou_web_e_docs = processar_query(query)
    
    st.subheader("Resposta")
    st.info(resposta)
    
    st.divider()
    st.subheader("Fontes")
            
    if usou_web_e_docs:
        st.success("Web e Documentos locais")
        for i, contexto in enumerate(contextos,1):
            if 'link' in contexto.keys():
                with st.expander(contexto.get('title', 'Resultado')):
                    st.write(f"Link: {contexto.get('link')}")
                    st.write(contexto.get('snippet', ''))
            else:
                with st.expander(f"Fonte {i}: {os.path.basename(contexto['path_origem'])}"):
                    st.write(f"**Similaridade:** {contexto['similaridade']:.0%}")
                    if contexto.get('num_pagina'):
                        st.write(f"**Página:** {contexto['num_pagina']}")
                    st.divider()
                    st.text_area("Conteúdo", contexto['conteudo'], height=150, key=f"ctx_{i}")
    elif usou_web:
        st.success("Web")
        for contexto in contextos:
            with st.expander(contexto.get('title', 'Resultado')):
                st.write(f"Link: {contexto.get('link')}")
                st.write(contexto.get('snippet', ''))
    else:
        st.success("Documentos locais")
        for i, contexto in enumerate(contextos, 1):
            with st.expander(f"Fonte {i}: {os.path.basename(contexto['path_origem'])}"):
                st.write(f"**Similaridade:** {contexto['similaridade']:.0%}")
                if contexto.get('num_pagina'):
                    st.write(f"**Página:** {contexto['num_pagina']}")
                st.divider()
                st.text_area("Conteúdo", contexto['conteudo'], height=150, key=f"ctx_{i}")



