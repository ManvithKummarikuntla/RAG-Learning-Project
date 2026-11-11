from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter,RecursiveCharacterTextSplitter

data = PyPDFLoader("C:/Users/my pc/Desktop/RAG_Proj/Document loaders/2410.15944v1.pdf")
docs = data.load()

splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10

)

chunks = splitter.split_documents(docs)


print(chunks[0].page_content)