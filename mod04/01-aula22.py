from langchain_community.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()

search_result = ddg_search.run('Qual é o Óleo usado no Palio Flex 1.0 2007?')
print(search_result)
