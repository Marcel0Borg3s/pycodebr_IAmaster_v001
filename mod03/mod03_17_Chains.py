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

# Construir a cadeia (chain) usando pipe (|) — mod03 Runnable Sequence
runnable_sequence = (
    PromptTemplate.from_template(
        'Me fale sobre o carro {carro}.',
    )
    | model
    | StrOutputParser()
)

response = runnable_sequence.invoke({'carro': 'Punto 2013'})

print(response)
