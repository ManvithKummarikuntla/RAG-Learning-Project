from langchain_community.document_loaders import ArxivLoader

loader = ArxivLoader(
    query="Retrieval Augmented Generation",
    load_max_docs=2
)

docs = loader.load()

for doc in docs:
    print(doc.metadata)
    print(doc.page_content[:500])