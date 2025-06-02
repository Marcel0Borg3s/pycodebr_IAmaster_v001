import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Modelo
model = ChatOpenAI(model_name='gpt-4', openai_api_key=OPENAI_API_KEY)

pdf_path = 'Manual Acer Aspire 5'
loader = PyPDFLoader(pdf_path)

docs = loader.load()

# Quebrar em chunks > from langchain_text_splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, #aqui é o tamanho da quebra em tokens
    chunck_overlap=200, #e aqui já é o tamanho do overlap 
)
chunks = text_splitter.split_documents(
    documents=docs,
)

embedding = OpenAIEmbeddings() #Se não indicar o model ele vai usar o padrão
# Criar o Vector Store para utilizar os chunks criados acima
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name='laptop_manual',

)

retriever = vector_store.as_retriever()

result = retriever.invoke(
    'Qual sistema operacional vem instalado?'
)
print(result)

