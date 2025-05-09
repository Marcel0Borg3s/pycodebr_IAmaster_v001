import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI

# Carrega variáveis de ambiente
load_dotenv()

# Garante que a variável esperada pelo LangChain esteja no ambiente
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("A variável OPENAI_API_KEY não foi carregada. Verifique o arquivo .env")

os.environ["OPENAI_API_KEY"] = api_key

model = ChatOpenAI(
    model="gpt-3.5-turbo",
)
messages = [
    {'role': 'system', 'content': 'Você é uma expert em Python e tem um conhecimento profundo sobre o assunto.'},
    {'role': 'user', 'content': 'Quem criou o Python?'},
]

response = model.invoke(messages)

print(response.content)