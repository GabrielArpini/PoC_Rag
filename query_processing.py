from typing import List, Dict, Any, Tuple

import psycopg2
import os
from langchain_aws import BedrockEmbeddings, Bedrock 
from langchain_aws.prompts import PromptTemplate 

from db_utils import get_conn

conn = get_conn()
cur = conn.cursor()

def get_query_embedding(query: str) -> List[float]:
    raise NotImplementedError

def pesquisa_semantica(query_embedding: List[float], top_k: int = 3):
    raise NotImplementedError

def gerar_resposta(query:str, contextos):
    raise NotImplementedError

def processar_query(query:str, top_k: int = 3):
    raise NotImplementedError 

if __name__ == "__main__":
    print("Realiza teste.")


