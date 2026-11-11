from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta = True
)

docs = retriever.invoke("Large Language Models")

for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}:")
    print("Title:", doc.metadata.get("Title"))
    print("Metadata:", doc.metadata)
    print("-" * 50)

