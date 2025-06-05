import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(
    model='gpt-4',
)

persist_directory = 'db'
embedding = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name='manual_laptop',
)

retriever = vector_store.as_retriever()

systm_prompt = '''
Use o contexto abaixo para responder a pergunta do usuário.
Contexto: {context}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', systm_prompt),
        ('human', '{input}'),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)
chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

query = 'Qual é a resolução da tela do laptop?'
response = chain.invoke(
    {'input': query},
)

print(response)


