# PoC RAG + Embeddings
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.50.0+-red.svg)
![PostgreSQL](https://img.shields.io/badge/postgresql-14+-blue.svg)

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
TODO: INSERIR IMAGEM PROCESSAMENTO QUERY.

O processamento de query ocorre após o usuário definir uma query e apertar o botão `processar` na interface web. Este processamento ocorre no arquivo `query_processing.py` iniciando com a transformação da query em um embedding utilizando o modelo "Amazon Titan Text Embedding v2", a obtenção de embedding acontece na função `get_query_embedding()`. 
Os embeddings da query são então comparados com os embeddings do banco de dados, usufruindo das capacidades da extensão `pgvector` do `PostgreSQL` para realizar a comparação dos embeddings com base na distância de coseno entre os vetores, o qual é feito utilizando o operador `<=>` do `pgvector`, extraindo um valor de similaridade que é filtrado com base em um número mínimo de similaridade.
Após as etapas mencionadas, é feito uma simples condicional com o intuito de verificar se há algum resultado com valor de similaridade acima do mínimo ou não, caso não exista ocorre o `fallback para a web`, onde há a obtenção de contextos a partir de uma pesquisa simples na web, um `otimizador de query para pesquisa` foi desenvolvido para aumentar a eficácia da pesquisa, o que de fato mostrou um resultado positivo. A função `otimizar_prompt_web()` utiliza a query do usuário para então gerar uma query para pesquisa na internet e o modelo de LLM (`Claude Haiku`) é utilizado com um prompt específico para esta tarefa, o contexto é então utilizado para gerar o prompt para a LLM (`Claude Haiku`) gerar a resposta da query do usuário, no caso de uso de internet não é feito mais nada depois.
Para o caso de a pesquisa de similaridade semântica retornar algum valor com chunks dos documentos relevantes, o prompt para LLM é então gerado e uma resposta é feita, neste caso há uma verificação da resposta da LLM com o contexto fornecido, ambos são transformados em embeddings e é feito um cálculo de similaridade com base na distância dos vetores em coseno, caso ultrapasse um valor de mínimo acetitável, a resposta é aprovada para ser mostrada ao usuário, caso contrário ocorre um `fallback para a web` com o intuito de obter contextos para reforçar os já presentes melhorando (com maior probabilidade) a resposta da LLM posteriormente, a função `otimizar_prompt_web` é utilizada para realizar a pesquisa, após isso o prompt final é gerado e passado para a LLM gerar uma resposta finalizando o processamento de query.


### Interface Web 
A interface web utiliza a biblioteca `Streamlit`, a fim de facilitar a demonstração da pipeline RAG. Para a criação foi utilizado `vibe code` (Justificativa: O streamlit já possui diversos componentes já pré feitos, e como medida de usar o tempo como eficiência utilizei o vibe code para gerar o design da interface, limitando o modelo a usar apenas os recursos presentes na biblioteca). Há diversas implementações a serem feitas, as quais serão compartilhadas na seção `Próximos passos`.

# Tecnologias utilizadas 
- **Backend**: Python 3.12
- **LLM**: AWS Bedrock (Claude 3 Haiku)
- **Embeddings**: Amazon Titan Embed Text v2
- **Vector DB**: PostgreSQL + pgvector
- **Frontend**: Streamlit
- **Web Search**: DuckDuckGo

# Pré-requisitos
- Acesso ao serviço Bedrock da AWS
- Docker
- uv (gerenciamento de pacotes e versões)

# Como usar
Primeiro é necessário clonar o repósitorio:
```bash
git clone https://github.com/GabrielArpini/PoC_Rag.git
cd PoC_Rag 
```

Agora é preciso configuar as credenciais da sua conta AWS:
```bash
uv run aws configure 
```
Este comando irá solicitar algumas credenciais para usar os serviços da AWS, por favor preencha-os.

Após isso, é necessário iniciar o banco de dados com o Docker compose:
```bash 
docker compose up -d 
```
`-d` siginifica detach, assim não ocupará um terminal. 

Com o banco de dados funcionando, basta iniciar a interface web com:
```bash
uv run streamlit run web_page.py
```

E voilá, a interface web abrirá no seu navegador padrão e poderá utilizar as suas funcionalidades, como realizar uma pergunta ou fazer o upload de um arquivo!

# Próximos passos 

- Otimizar as chamadas de API da AWS, minimizando ao máximo os custos.
- Melhorar as interações com o banco de dados, a query de inserção e update realiza os updates partindo da premissa que 
se houver chunks com metadados iguais quer dizer que devem ser atualizados, não necessáriamente é o caso.
- Colocar opções na interface para manipular hiperparâmetros (MIN_SIMILARIDADE, min_similaridade_res, chunk_size, chunk_overlap)
- Melhorar a visualização de itens processados do banco de dados na interface web; 
- Possibilitar a exclusão de itens do banco de dados através da interface web; 
- Criar interface web para visualizar os embeddings presentes no banco de dados e estatísticas. 





