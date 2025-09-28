from typing import Optional, Dict, Any, List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document # Type annotation
import os 
import psycopg2 

from langchain_aws import BedrockEmbeddings


PDFS_PATH = 'data/pdfs'
TXTS_PATH = 'data/txts'


conn = conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME', 'rag_db'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'postgres'),
    host=os.getenv('DB_HOST', 'localhost'),
    port='5432'
)
cur = conn.cursor()

def carregar_documentos(diretorio: str, loader_cls: Any, tipo_arquivo: str) -> list[Document]:
    # TODO: Criar utils para checagem da existencia de DATA_PATH
    # e conteudos
    #
    loader = DirectoryLoader(
            diretorio, 
            glob=f"**/*.{tipo_arquivo}",
            loader_cls = loader_cls
            )
    return loader.load()

def chunk_document(documentos: list[Document]) -> list[Document]:
    # https://python.langchain.com/docs/how_to/recursive_text_splitter/

    splitter_texto = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len,
        is_separator_regex = False,
    )
    return splitter_texto.split_documents(documentos)

#TODO: Criar uma função para verificar se houve update no pdf/texto 
# por ora não há uma forma de atualizar o dados caso eles sejam atualizados.

def armazenar_db(chunks_tratados: List[Dict[str,Any]]) -> None:
    for chunk in chunks_tratados:
        cur.execute(
            """ 
            INSERT INTO docs (path_origem,num_pagina,indice_chunk,conteudo,embedding,modtempo)
            VALUES (%s,%s,%s,%s,%s,%s)
            """,
            (
                chunk['path_origem'],
                chunk['pag'],
                chunk['indice_chunk'],
                chunk['conteudo'],
                chunk['embedding'],
                chunk['modtempo']
            )
        )
    conn.commit()

# TODO: Essa funcao vai ser chamada a cada chunk, otimizar com batches posteriormente.
def get_embedding(texto:str):
    embedding = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0"
    )
    return embedding.embed_query(texto)


def processar_chunks_pdf(chunks, dados_chunk):
    ultima_pagina = None
    indice_chunk_atual = 0
    for chunk in chunks:
        conteudo = chunk.page_content
        metadados = chunk.metadata
        pagina_atual = metadados['page']
        if ultima_pagina == pagina_atual:
            indice_chunk_atual += 1
        else:
            indice_chunk_atual = 0 # Reseta o indice para pagina nova.
            ultima_pagina = pagina_atual
        
        embedding = get_embedding(conteudo)
        dados_chunk.append({
            "path_origem": metadados['source'],
            "conteudo": conteudo,
            "pag": pagina_atual,
            "indice_chunk": indice_chunk_atual,
            "embedding": embedding,
            "modtempo": metadados['moddate'] # Abreviacao para ultima modificacao/modificacao tempo.
        })

def processar_chunks_txt(chunks, dados_chunk):
    indice_chunk_atual = 0
    for chunk in chunks:
        conteudo = chunk.page_content
        metadados = chunk.metadata
        embedding = get_embedding(conteudo)

        dados_chunk.append({
            "path_origem": metadados['source'],
            "conteudo": conteudo,
            "pag": None,
            "indice_chunk": indice_chunk_atual,
            "embedding": embedding,
            "modtempo": None
        })
        indice_chunk_atual += 1

def main():
    pdfs = carregar_documentos(PDFS_PATH, PyPDFLoader, "pdf")
    txts = carregar_documentos(TXTS_PATH, TextLoader, "txt")
    chunks_txts = chunk_document(txts)
    chunks_pdfs = chunk_document(pdfs)

    dados_chunk = [] # Lista que armazenará o dict com informações processadas.
    
    processar_chunks_pdf(chunks_pdfs,dados_chunk)
    processar_chunks_txt(chunks_txts, dados_chunk)

    # Agora é possível armazenar as informações na db.
    armazenar_db(dados_chunk)
    


            
    
if __name__ == '__main__':
    print("Começando...")
    main()
    print("Concluido!")

    #cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema ='public'")
    #results = cur.fetchall()
    #for row in results:
        #print(row)
