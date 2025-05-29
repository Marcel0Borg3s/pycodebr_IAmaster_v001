from langchain_community.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()

search_result = ddg_search.run('Quem criou o LINUX')
print(search_result)
