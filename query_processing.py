from typing import List, Dict, Any, Tuple

import psycopg2
import os
from langchain_aws import BedrockEmbeddings, BedrockLLM 
from langchain.prompts import PromptTemplate 
import boto3 

from db_utils import get_conn

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_community.tools import DuckDuckGoSearchResults

# https://python.langchain.com/docs/integrations/llms/bedrock/
class BedrockAsyncCallbackHandler(AsyncCallbackHandler):
    # Async callback handler that can be used to handle callbacks from langchain.

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        reason = kwargs.get("reason")
        if reason == "GUARDRAIL_INTERVENED":
            print(f"Guardrails: {kwargs}")

# Valor minimo aceito de similaridade para considerar o contexto bom. 
MINIMO_SIMILARIDADE = 0.4

conn = get_conn()
cur = conn.cursor()


def get_query_embedding(query: str) -> List[float]:
    embedding = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0"
    )
    return embedding.embed_query(query)

def pesquisa_semantica(query_embedding: List[float], top_k: int = 3) -> List[Dict[str,Any]]:
    pgvector_query = """
        SELECT 
            path_origem,
            num_pagina,
            indice_chunk,
            conteudo,
            1 - (embedding <=> %s::vector) as similaridade 
        FROM docs
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    try:
        cur.execute(pgvector_query,(query_embedding,query_embedding, top_k))
        resultados = cur.fetchall()

        lista_resultados = [
            {
                "path_origem": linha[0],
                "num_pagina": linha[1],
                "indice_chunk": linha[2],
                "conteudo": linha[3],
                "similaridade": linha[4]
            }
            for linha in resultados
            if linha[4] >= MINIMO_SIMILARIDADE
        ]
        return lista_resultados
    except Exception as e:
        print(f"Erro durante pesquisa semantica: {e}")
        return []


def gerar_resposta(query:str, contextos: List[Dict[str,Any]], usou_web:bool = False):
    if not usou_web:
        info_contextos = "\n".join(
                f"Origem: {contexto['path_origem']}, pag: {contexto['num_pagina'] or 'Sem pagina'}, chunk: {contexto['indice_chunk']}\nConteudo:\n{contexto['conteudo']}" 
            for contexto in contextos
        )

    else:
        info_contextos = "\n".join(
            f"Link: {contexto['link']}, titulo: {contexto['title']} \nSnippet:\n{contexto['snippet']}" 
            for contexto in contextos
        )
    template_prompt = PromptTemplate(
        input_variables = ['query','contexto'],
        template= """
        Voce e um assistente que responde as perguntas do usuario e utiliza o contexto fornecido
        como fonte de verdade. Na sua resposta, deve colocar a origem do conteudo que utilizou(utilize as informacoes do contexto, se for um link, coloque o link utilizado).
        Se o contexto conter chunk, pagina e origem, coloquê-os na sua resposta.
        Utilize o contexto como base para a sua resposta da pergunta apenas, e como ja dito deixe uma secao para as fontes de onde tirou a informação.
        Query do usuario: {query}
        Contexto: 
        {contexto}
        Query do usuario: {query}
        Sua resposta:
        """
    )

    prompt = template_prompt.format(query=query,contexto=info_contextos) 
    resposta = get_resposta_modelo(prompt)

    return resposta 

def get_resposta_modelo(prompt: PromptTemplate):
    client = boto3.client("bedrock-runtime")
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    conversa = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    try:
        response = client.converse(
            modelId=model_id,
            messages=conversa,
            inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
        )
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text 
    except Exception as e:
        print(f"Um erro aconteceu durante criacao da resposta: {e}")


        
def buscar_na_web(query:str):
    """
    A principio assumo que os resultados da web já sao relevantes, caso resultado final 
    seja ruim, seguir o TODO.
    TODO: Implementar calls de embedding para comparar os vetores da query com snippets da internet. 
    """
    search = DuckDuckGoSearchResults(output_format="list")
    return search.invoke(query)


def processar_query(query:str, top_k: int = 5):
    query_embedding = get_query_embedding(query)
    contextos = pesquisa_semantica(query_embedding,top_k) 
    usou_web = False 
    if not contextos:
        print("Nao foi encontrado um contexto no(s) texto(s), pesquisando na web...")
        usou_web = True
        contextos = buscar_na_web(query)

    resposta = gerar_resposta(query,contextos,usou_web)
    print(resposta)
    return resposta, contextos, usou_web 
    
if __name__ == "__main__":
    print("Realiza teste.")
    #resposta, _, _, _ = processar_query("O que é um processo fonatorio?")
    #print(resposta)
