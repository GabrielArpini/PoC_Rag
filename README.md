# PoC RAG

## Introdução 
O presente projeto é uma demonstração de uma PoC de um assistente de perguntas e respostas que utiliza documentos como fonte de verdade para o processo seletivo da Kairos Lab.

## Primeiras intuições
Como foi citado o uso de AWS internamente pela empresa, darei prioridade aos recursos presentes na AWS para o desenvolvimento deste projeto, sendo o AWS Bedrock o item principal da pipeline.

Este projeto deve ser dividido em duas partes:

### Pré-processamento
Utilizar os documentos fornecidos, separá-los em chunks(os indíces podem ser usados para encontrar a fonte em que o modelo se baseou para uma resposta posteriormente), criar os embeddings e armazená-los em um vectorDB, provavelmente ChromaDB ou PGVector + PostgreSQL.

### Processamento de pergunta e resposta
Após o envio de uma pergunta pelo usuário, a mesma será processada em uma pesquisa de similaridade na VectorDB, a qual retornará chunks relevantes e a pergunta será aprimorada antes de enviada para a LLM, caso não haja nada relevante na VectorDB, a pergunta será enviada diretamente para a LLM.

### Fine tuning
Seguir as instruções apresentadas no documento.

## Opcionais (prováveis de serem aplicados)
- Interface web;
- Web search;
- RLHF (muito ousado, se sobrar bastante tempo, sistema de feedback de respostas);
- etc..


## Filosofia
- Código limpo;
- Código simples, sem complexidade desnecessária;
- Type annotations para legibilidade;
- Eficiência de custo, priorizar soluções open sources que maximizam eficiência e minimiza custos;
- Transparência através de documentação clara e comentários claros nos códigos.

# Tecnologias
- Linguagem: Python 3.8+
- Ferramentas: boto3, LangChain, psycopg2(pgvector);
- VectorDB: PostgreSQL + pgvector, ChromaDB (alternativa)
- AWS: Bedrock embeddings (Titan) e inferência(modelo a definir)
- Documentos: PDFs e .txts

# How to use 
First run and setup the AWS credentials, the user only needs BedrockFullCredentials

´´´bash
uv run aws configure
´´´

