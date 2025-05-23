import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Modelo
model = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)

# Ferramenta Wikipedia
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(lang='pt'),
)


# Criar um Agente

agent_executor = initialize_agent(
    tools=[wikipedia_tool],
    llm=model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Prompt
prompt_template = PromptTemplate(
    input_variables=['query'],
    template="""
    Pesquise na web sobre {query} e forneça um resumo sobre o assunto
    """,
)

query = 'Quem criou o Python?'
prompt = prompt_template.format(query=query)

# Rodar Agent
response = agent_executor.invoke({'input': prompt})
print(response.get('output'))
