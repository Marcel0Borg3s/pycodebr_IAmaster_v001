import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate


# Load environment variables from .env file with dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create an OpenAI client
model = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)


# Classificar o input do User
classification_chain = (
    PromptTemplate.from_template(
        """
        Classifique o input do Usuário em um dos seguintes setores:
        - Financeiro
        - Suporte Técnico
        - Outras informações
        
        Pergunta: {pergunta}
        """
    )
    | model
    | StrOutputParser()
)

financial_chain = (
    PromptTemplate.from_template(
        """
        Você é um especialista financeiro.
        Sempre responda as perguntas começando com um "Bem-vindo ao setor Financeiro".
        Responda à pergunta do Usuário: {pergunta}
    
        """
    )
    | model
    | StrOutputParser()
)

tech_support_chain = (
    PromptTemplate.from_template(
        """
        Vocé é um especialista em Suporte Técnico.
        Sempre responda as perguntas com um "Bem-vindo ao Suporte Técnico".
        Responda à pergunta do Usuário: {pergunta}
    
        """
    )
    | model
    | StrOutputParser()
)

other_info_chain = (
    PromptTemplate.from_template(
        """
        Vocé é um Assistente de informações Gerais.
        Sempre responda as perguntas com um "Bem-vindo a Central de Informações".
        Responda à pergunta do Usuário: {pergunta}
    
        """
    )
    | model
    | StrOutputParser()
)


def route(classification):
    classification = classification.lower()
    if 'financeiro' in classification:
        return financial_chain
    elif 'suporte técnico' in classification:
        return tech_support_chain
    else:
        return other_info_chain


pergunta = input('O que deseja saber? ')
classification = classification_chain.invoke({'pergunta': pergunta})
response_chain = route(classification=classification)

response = response_chain.invoke({'pergunta': pergunta})

print(response)
