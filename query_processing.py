from typing import List, Dict, Any, Tuple

import psycopg2
import os
from langchain_aws import BedrockEmbeddings, BedrockLLM 
from langchain.prompts import PromptTemplate 
import boto3 
import numpy as np 

from db_utils import get_conn

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_community.tools import DuckDuckGoSearchResults

#

# Valor minimo aceito de similaridade para considerar o contexto bom. 
MINIMO_SIMILARIDADE = 0.30

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
    info_contextos = ""
    for contexto in contextos:
        if 'link' not in contexto.keys(): # link so existe se pesquisou na web 
            info_contextos += f"\nOrigem: {contexto['path_origem']}, pag: {contexto['num_pagina'] or 'Sem pagina'}, chunk: {contexto['indice_chunk']}\nConteudo:\n{contexto['conteudo']}" 

        else: 
            info_contextos += f"\nLink: {contexto['link']}, titulo: {contexto['title']} \nSnippet:\n{contexto['snippet']}" 
            

    template_prompt = PromptTemplate(
        input_variables = ['query','contexto'],
        template= """
        Voce e um assistente que responde as perguntas do usuario e utiliza o contexto fornecido
        como fonte de verdade. Na sua resposta, deve colocar a origem do conteudo que utilizou(utilize as informacoes do contexto, se for um link, coloque o link utilizado).
        Se o contexto conter chunk, pagina e origem, coloquê-os na sua resposta.
        Utilize o contexto como base para a sua resposta da pergunta apenas, e como ja dito deixe uma secao para as fontes de onde tirou a informação.
        Caso tenha um link, verifique se o link em si possui similaridade com a query do usuario.
        Sua resposta deve conter no maximo 512 tokens.
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

def otimizar_prompt_web(user_query) -> PromptTemplate:
    """
    Função feita para os fall backs para web, com o intuito de criar um prompt 
    com base na query do usuario e retornar um prompt otimizado para pesquisa na web.
    """
    template_prompt = PromptTemplate(
        input_variables = ['query','contexto'],
        template= """
        Você é um especialista em motores de busca. Sua tarefa é reescrever a 'Pergunta Original do Usuário' a seguir em uma 'Consulta Otimizada para a Web'.
        A consulta otimizada deve:
        - Ser curta e focada nas palavras-chave mais importantes.
        - Capturar a intenção principal da pergunta original.
        - Ser ideal para obter resultados relevantes em um motor de busca como o Google.
        REGRAS:
        - NÃO responda à pergunta do usuário.
        - NÃO inclua nenhuma palavra antes ou depois da consulta otimizada (sem preâmbulos como 'Aqui está a consulta:').
        - A saída deve ser apenas a nova consulta e nada mais.

        Pergunta Original do Usuário: '{query}'

        Consulta Otimizada para a Web:
        """
    )

    prompt = template_prompt.format(query=user_query)
    return prompt 
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
    (Encontrar possiveis solucoes para a questao da diferenca da query para o snippet 
    Por exemplo: Quem foi a primeira pessoa a pisar na lua?
    Resposta: Neil Armstrong 
    Neste exemplo o embedding da pergunta não necessariamente possui uma similaridade semantica 
    grande com a resposta, e este tipo de problema pode resultar em erros durante a procura de snippets.
    """
    search = DuckDuckGoSearchResults(output_format="list")
    return search.invoke(query)


def processar_query(query:str, top_k: int = 5, min_similaridade_res=0.6):
    query_embedding = get_query_embedding(query)
    contextos = pesquisa_semantica(query_embedding,top_k) 
    usou_web = False
    usou_web_e_docs = False
    if not contextos:
        print("Nao foi encontrado um contexto no(s) texto(s), pesquisando na web...")
        usou_web = True
        query_web = get_resposta_modelo(otimizar_prompt_web(query))
        contextos = buscar_na_web(query_web)

    resposta = gerar_resposta(query,contextos,usou_web)

    # Comparar a respota com os contextos existentes.
    # Caso ja tenha pesquisado na internet, nao fara nada.
    # porem, caso tenha contexto apenas dos docs eles sejam considerados nao suficientes
    # adicionara contextos da web.
    if not usou_web:
        contextos_embeddings = [get_query_embedding(contexto['conteudo']) for contexto in contextos]
        resposta_embeddings = get_query_embedding(resposta)
        # Calcular similaridade de coseno manualmente ao inves de criar tabela temporaria
        # e usar a capacidade do pgvector para isso. 
        contextos_np = np.array(contextos_embeddings)
        resposta_np = np.array(resposta_embeddings)
        produto = np.dot(contextos_np,resposta_np)

        magnitude_cont = np.linalg.norm(contextos_np,axis=1) #axis e 1 para realizar o calculo de cada vetor ao inves da matriz
        magnitude_res = np.linalg.norm(resposta_np)

        similaridade = np.max(produto / (magnitude_cont*magnitude_res))
        if similaridade < min_similaridade_res:
            query_para_web_otimizada = get_resposta_modelo(otimizar_prompt_web(query))
            contextos += buscar_na_web(query_para_web_otimizada)
            usou_web_e_docs = True
            resposta = gerar_resposta(query,contextos,usou_web)
    print(resposta)
    return resposta, contextos, usou_web, usou_web_e_docs 
    
if __name__ == "__main__":
    print("Realiza teste.")
    #resposta, _, _, _ = processar_query("O que é um processo fonatorio?")
    #print(resposta)
