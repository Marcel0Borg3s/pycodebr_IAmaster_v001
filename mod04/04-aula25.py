import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modelo
model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)



prompt = '''
Como um assistente financeiro pessoal, que deverá responder perguntas dando dicas financeiras de investimento,
Responder todas as perguntas de forma direta e objetiva,
Perguntas: {query}
'''

prompt_template = PromptTemplate(
    template=prompt,
    input_variables=["query"]
)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name="Python REPL",
    description="""
    Um Shell Python, use isso para executar códigos Python, execute apenas codigos Python válidos
    Se necessário obter um retorno, use a função 'print(...)'
    Use para fazer cálculos finaceiros, responder as perguntas e dar dicas financeiras
    """,
    func=python_repl.run
)

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    description="""
Um Shell Python. Use isso para executar códigos Python válidos.
Se necessário obter um retorno, use a função "print(...)"
Use para fazer cálculos financeiros, responder perguntas e dar dicas financeiras.
""",
    func=search.run
)

# Criar um Agente de Reação / Agent React
react_instructions = hub.pull('hwchase17/react') #isso é um modelo pronto de Agent React

tools = [python_repl_tool, duckduckgo_tool]

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_instructions,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

question = '''
Minha renda é de R$4500,00 por mês, somando despeas fixas no valor de R$1500,00 e variáveis de R$2900,00.
Quais dicas você me daria para investir a diferença e ter uma reserva de emergência?
'''

output = agent_executor.invoke(
    {'input': prompt_template.format(query=question)}
)

print(output.get('output'))