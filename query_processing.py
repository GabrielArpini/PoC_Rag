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
    """
    Gera embeddings da string input.

    Args:
        query (str): String a se obter embeddings.
    Returns:
        List[float]: Lista com os embeddings.
    """

    embedding = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0"
    )
    return embedding.embed_query(query)

def pesquisa_semantica(query_embedding: List[float], top_k: int = 3) -> List[Dict[str,Any]]:
    """
    Utiliza a extensão 'pgvector' do banco de dados para comparar o embedding input com os 
    embedings do banco de dados através do cálculo de similaridade por coseno (<=>).

    Args:
        query_embedding (List[float]): Lista com os embeddings de entrada para serem comparados com o banco de dados.
        top_k (int): Quantidade de itens a retornar (caso existam). Default: 3.
    Returns:
        List[Dict[str, Any]]: Lista com os resultados, vazia se não for achado nada.
    Raises:
        Exception: Caso ocorra um erro durante a query é rotornado uma lista vazia.
    """

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
    """
    Utiliza a query em conjunto com os contextos para gerar uma resposta do modelo através do Prompt gerado.
    Não gera a resposta em sí, a função 'get_resposta_modelo' que interage com a API para obter a resposta da LLM.

    Args:
        query (str): string a ser usada como query do usuário.
        contextos (List[Dict[str,Any]]): lista com os contextos a serem apresentados para a LLM. 
        usou_web (bool): booleana para definir se foi usado web ou não para obter os contextos. 
    Returns:
        resposta (str): resposta da LLM.
    """
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

def otimizar_prompt_web(user_query: str) -> PromptTemplate:
    """
    Função feita para os fall backs para web, com o intuito de criar um prompt 
    com base na query do usuario e retornar um prompt otimizado para pesquisa na web.

    Args: 
        user_query (str): string a ser otimizada para web.
    Returns:
        PromptTemplate: Prompt a ser passado para a LLM.
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
    """
    Iterage com a API do bedrock e o modelo claude-3-haiku para gerar uma resposta com base do prompt 
    de parâmetro.

    Args:
        prompt (PromptTemplate): prompt a ser passado para a LLM.
    Returns:
        response_text (str): Resposta do modelo.
    Raises:
        Exception: Caso ocorra um erro durante a chamada da API é printado uma mensagem de erro
        e retorna uma string vazia.
    """
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
        return ""


        
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

    Args:
        query (str): query de pesquisa para a web. 
    Returns:
        List[Dict[str,str]]: Retorna uma lista de dicionários com as informações da pesquisa(snippet, link, etc)  
    """
    search = DuckDuckGoSearchResults(output_format="list")
    return search.invoke(query)


def processar_query(query:str, top_k: int = 5, min_similaridade_res=0.6):
    """
    Função principal que junta todas as funcionalidades, desde a obtenção de embeddings até gerar as respostas e fallbacks.
    Args:
        query (str): query do usuário.
        top_k (int): máximo de valores a retornar da busca por similaridade semantica.
        min_similaridade_res (float): Minimo de similaridade aceita da resposta do modelo.
    Returns:
        resposta (str): Resposta do modelo.
        contextos (List[Dict[str,Any]]): Lista de dicionários com todos os contextos utilizados para a resposta.
        usou_web (bool): Se usou a web a priori ou não.
        usou_web_e_docs (bool): se precisou usar a web após gerar a primeira resposta.
    """
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
    # porem, caso tenha contexto apenas dos docs e eles sejam considerados nao suficientes
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
