import streamlit as st
from query_processing import processar_query

st.title("Assistente com RAG")

query = st.text_input("Faça uma pergunta:")

if st.button("Processar") and query:
    with st.spinner("Processando..."):
        resposta, contextos, usou_web = processar_query(query)  # modify to return these
    
    st.subheader("Resposta:")
    st.write(resposta)
    
    st.subheader("Fontes:")
    if usou_web:
        st.info("Realizou pesquisa na web.")
        for contexto in contextos:
            with st.expander(f"{contexto.get('title', 'Resultado da Web')}"):
                st.write(f"Link: {contexto.get('link', 'Nenhum')}")
                st.write(contexto.get('snippet', ''))
    else:
        st.info("Usou documentos locais.")
        for contexto in contextos:
            with st.expander(f"Fonte: {contexto['path_origem']}"):
                st.write(f"Similaridade: {contexto['similaridade']:.2f}")
                st.write(contexto['conteudo'])
