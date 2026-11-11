from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(
    top_k_results=3,
    lang='en'
)

#docs=retriever.load()
docs=retriever.invoke("What is the latest research on generative AI?")

for doc in docs:
    print("Content:", doc.page_content[:500])
    print("Metadata:", doc.metadata)
    print("-" * 50)
