#we can save file as .txt or .pdf

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="", 
    chunk_size=15, 
    chunk_overlap=2    
)


data=TextLoader("C:/Users/my pc/Desktop/RAG_Proj/Document loaders/test.txt")

docs=data.load()

chunks=splitter.split_documents(docs)

for i in chunks:

    print(i.page_content)
    print("---------------------------------------------------")