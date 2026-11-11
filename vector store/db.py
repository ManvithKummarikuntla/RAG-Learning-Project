from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]


vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    persist_directory = "chroma-db"
)

result = vectorstore.similarity_search("What is used for data analysis",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vectorstore.as_retriever()
docs = retriever.invoke("What is python")

for d in docs:
    print(d)