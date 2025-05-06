from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Carrega o PDF
loader = PyPDFLoader("arquivo.pdf")
documentos = loader.load()

# Carrega o modelo
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Cria a cadeia de QA
chain = load_qa_chain(llm, chain_type="stuff")

# Faz perguntas específicas
nome = chain.run(input_documents=documentos, question="Qual o nome do solicitante?")
cnpj = chain.run(input_documents=documentos, question="Qual o CNPJ mencionado no documento?")

print("Nome:", nome)
print("CNPJ:", cnpj)
