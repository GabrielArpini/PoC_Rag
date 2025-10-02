# PoC RAG + Embeddings

## Introdução
Pipeline RAG com embeddings desenvolvida para o processo seletivo da Kairos Lab, em conjunto com um notebook para o fine tunin do modelo `Gemma 3 270m`. A pipeline principal é integrada com a interface web que utiliza `streamlit` para facilitar a demonstração de suas funcionalidades.

## Como a pipeline RAG funciona?
O desenvolvimento da pipeline RAG foi separado em dois componentes principais: pre processamento e processamento de query.

### Pre-processamento 
O pré processamento(`pre_processing.py`) é um componente desenvolvido para o processamento de arquivos pdf e txt, o qual satisfaz o seguinte diagrama simplificado

#TODO: inserir diagrama 

A partir da interface web, pode-se inserir múltiplos arquivos (pdf ou txt), os quais são usados para iniciar o processo de pré processamento conforme a imagem. Primeiro é feito uma simples extração da extensão do arquivo inserido, para então utilizar o `loader` da biblioteca `langchain` correspondente a extensão extraida (`PyPDFLoader` ou `TextLoader`), a função que realiza esta tarefa é a `processar_item_unico()`. Após o carregamento do arquivo especificado é feito a separação dos chunks que utiliza a função `RecursiveCharacterTextSplitter` da biblioteca `langchain`, após isso se utiliza uma variável `dados_chunk` para armazenar o índice do chunk, página (se houver), conteúdo, embedding do conteúdo, tempo da última modificação (se houver) e caminho do arquivo, finalizando o processamento do item, que ocorre iterativamente a cada item do upload de arquivos.

Com os arquivos já processados é feito uma verificação da existência de `órfãos` no banco de dados. Um órfão ocorre quando se tenta realizar um upload de um mesmo arquivo que já foi armazenado no banco de dados, mas que possui uma quantidade menor de  páginas por exemplo, neste caso para maior eficiência do espaço é feito a remoção de todos os órfaos do banco de dados com base nos arquivos que estão sendo processados para armazenamento no banco de dados, esta tarefa é realizada pela função `check_db_orfaos()`.

Por fim, é feito a inserção dos dados dos chunks no banco de dados, utilizando uma query SQL que também permite a atualização de chunks que foram modificados, maximizando a eficiência de processamento de arquivos.


### Processamento de query 

O processamento de query ocorre após o usuário definir uma query e apertar o botão `processar` na interface web. Este processamento ocorre no arquivo `query_processing.py` iniciando com a transformação da query em um embedding utilizando o modelo "Amazon Titan Text Embedding v2", a obtenção de embedding acontece na função `get_query_embedding()`. 
Os embeddings da query são então comparados com os embeddings do banco de dados, usufruindo das capacidades da extensão `pgvector` do `PostgreSQL` para realizar a comparação dos embeddings com base na distância de coseno entre os vetores, o qual é feito utilizando o operador `<=>` do `pgvector`



