import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
)   # TextLoader ou CSVLoader


# Load environment variables from .env file with dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create an OpenAI client
model = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)


loader = PyPDFLoader(r'E:\RPA\Arquivos_teste\CCT-modelo.pdf')
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=['context', 'question'],
    template="""Responda as perguntas apenas com base nas informações fornecidas em {context}. 
    Não utilizar informações externas para responder as perguntas, 
    se a pergunta não for relacionada aos dados fornecidos, responder como "Essas perguntas não se referem aos dados fornecidos".
    Pergunta: {question}
    """,
)

chain = prompt_base_conhecimento | model | StrOutputParser()

response = chain.invoke(
    {
        'context': '\n'.join(
            doc.page_content for doc in documents
        ),  # aqui para passar por todo texto conteúdo do arquivo lido
        'question': input('O que deseja saber? '),
    }
)
print(response)
