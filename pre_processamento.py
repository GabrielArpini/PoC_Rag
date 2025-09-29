from typing import Optional, Dict, Any, List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document # Type annotation
import os 
import psycopg2 
from langchain_aws import BedrockEmbeddings

from db_utils import get_conn 

PDFS_PATH = 'data/pdfs'
TXTS_PATH = 'data/txts'


conn = get_conn()
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
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        is_separator_regex = False,
    )
    return splitter_texto.split_documents(documentos)

#TODO: Criar uma função para verificar se houve update no pdf/texto 
# por ora não há uma forma de atualizar o dados caso eles sejam atualizados.

def armazenar_db(chunks_tratados: List[Dict[str,Any]]) -> None:
    """
    Função de extrema importância, uma vez que deve detectar mudanças nos pdfs/textos e tratá-las.
    Exemplos: O pdf é o mesmo, mas houve uma mudança em uma seção dele;
    Foi deletado uma página do pdf;
    Foi adicionado uma página ao pdf.

    Os statements SQL abaixo utilizam mecanismos do PostgreSQL para verificar estes casos, a fim de 
    lidar com o maior número possível de casos.
    
    A medida pensada foi de excluir primeiro e depois inserir/atualizar, primeiro se exclui da tabela o conteudo que
    foi excluido do texto/pdf depois se realiza um checking de atualizações no conteúdo ou novos conteúdos.

    # TODO: implementar logica para exclusão dos conteudos que foram removidos do pdf/texto.
    # TODO: implementar forma de criar os embeddings apenas quando necessário, assim 
    # as funções processar_chunks_* não precisarão de processar os embeddings.
    """
    for chunk in chunks_tratados:
        cur.execute(
            """ 
            INSERT INTO docs (path_origem,num_pagina,indice_chunk,conteudo,embedding,modtempo)
            VALUES (%s,%s,%s,%s,%s,%s)
            
            ON CONFLICT (path_origem, num_pagina, indice_chunk) DO UPDATE
            SET 
                conteudo = EXCLUDED.conteudo, -- EXCLUDED é uma tabela com os valores que iriam entrar mas que foram barrados.
                embedding = EXCLUDED.embedding, 
                modtempo = EXCLUDED.modtempo 
            WHERE
                docs.conteudo IS DISTINCT FROM EXCLUDED.conteudo AND 
                docs.embedding IS DISTINCT FROM EXCLUDED.embedding AND 
                docs.modtempo IS DISTINCT FROM EXCLUDED.modtempo
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


def processar_chunks_pdf(chunks, dados_chunk):
    ultima_pagina = None
    indice_chunk_atual = 0
    embedding = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0"
    )
    for chunk in chunks:
        conteudo = chunk.page_content
        metadados = chunk.metadata
        pagina_atual = metadados['page']
        if ultima_pagina == pagina_atual:
            indice_chunk_atual += 1
        else:
            indice_chunk_atual = 0 # Reseta o indice para pagina nova.
            ultima_pagina = pagina_atual
        
        conteudo_embedding = embedding.embed_query(conteudo)
        dados_chunk.append({
            "path_origem": metadados['source'],
            "conteudo": conteudo,
            "pag": pagina_atual,
            "indice_chunk": indice_chunk_atual,
            "embedding": conteudo_embedding,
            "modtempo": metadados.get('moddate') # Abreviacao para ultima modificacao/modificacao tempo. get para retornar None se nao existe.
        })

def processar_chunks_txt(chunks, dados_chunk):
    indice_chunk_atual = 0
    embedding = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0"
    ) 
    for chunk in chunks:
        conteudo = chunk.page_content
        metadados = chunk.metadata
        conteudo_embedding = embedding.embed_query(conteudo)

        dados_chunk.append({
            "path_origem": metadados['source'],
            "conteudo": conteudo,
            "pag": None,
            "indice_chunk": indice_chunk_atual,
            "embedding": conteudo_embedding,
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
