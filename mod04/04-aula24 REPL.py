import os
from langchain.agents import Tool
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Modelo
model = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)

# Ferramenta REPL
python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Um Shell Python, use isso para executar códigos Python, execute apenas codigos Python válidos, Se necessário obter um retorno, use a função "print(...)"',
    func=python_repl.run,
)


# Criar um Agente

agent_executor = create_python_agent(
    llm=model,
    tool=python_repl_tool,
    # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Prompt
prompt_template = PromptTemplate(
    input_variables=['query'],
    template="""
    Resolva o calculo {query} 
    """,
)

query = 'Calcule uma comissão de 15 porcento sobre o valor de imposto de 18 porcento de R$ 1000'
prompt = prompt_template.format(query=query)

# Rodar Agent
response = agent_executor.invoke(prompt)
print(response.get('output'))
