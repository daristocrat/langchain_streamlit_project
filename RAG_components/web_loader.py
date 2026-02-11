from langchain_community.document_loaders import WebBaseLoader
url='https://en.wikipedia.org/wiki/A_Knight_of_the_Seven_Kingdoms_(TV_series)'
loader=WebBaseLoader(url)

docs=loader.load()

print(len(docs))

print(docs[0].page_content)
