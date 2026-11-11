from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma.from_documents(docs,embeddings)#persist_directory="chroma-db")



similarity_retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3 }
)


print("------------------------------Similarity Search Results------------------------------")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")
for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k" : 3}
)

print("\n=============== MMR Search Results ===============\n")
mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)