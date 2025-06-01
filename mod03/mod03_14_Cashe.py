import os
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_openai import OpenAI, ChatOpenAI

# InmMemory ficara temporário (como memória RAM)
# SQLite ficara permanentemente
from langchain_community.cache import InMemoryCache, SQLiteCache

# Carrega variáveis de ambiente
load_dotenv()

# Garante que a variável esperada pelo mod03 esteja no ambiente
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError(
        'A variável OPENAI_API_KEY não foi carregada. Verifique o arquivo .env'
    )

os.environ['OPENAI_API_KEY'] = api_key

set_llm_cache(SQLiteCache(database_path='openai_cache.db'))

model = ChatOpenAI(
    model='gpt-3.5-turbo',
)
messages = [
    {
        'role': 'system',
        'content': ' Vocé é uma expert linkedin e tem um conhecimento profundo sobre o assunto.',
    },
    {
        'role': 'user',
        'content': 'Me de um texto para postar meu divulgar que estou criando esse bot IA no meu projeto de estudo de IA com mod03,'
        ' no curso de IA Master da PycodeBR',
    },
]

response = model.invoke(messages)

print(response.content)
