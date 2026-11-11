
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="BookMind AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 BookMind AI")
st.caption("Upload a book and interact with its knowledge using RAG.")


# --------------------------------------------------
# 2. INITIALIZE SESSION STATE
# --------------------------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# 3. API CONFIGURATION
# --------------------------------------------------

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()


@st.cache_resource
def initialize_models():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return embeddings, llm


embeddings, llm = initialize_models()


# --------------------------------------------------
# 4. PROMPT TEMPLATE
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI book assistant.

Answer the user's question using ONLY the
provided document context.

If the answer is not present in the context,
say:
"I could not find the answer in the document."

Do not make up information.

Mention relevant page numbers when available.
"""
        ),
        (
            "human",
            """Context:

{context}

Question:

{question}
"""
        )
    ]
)


# --------------------------------------------------
# 5. SIDEBAR - DOCUMENT UPLOAD
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Upload your book")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    process_button = st.button(
        "⚡ Process Document",
        use_container_width=True,
        disabled=uploaded_file is None
    )

    if uploaded_file is not None:
        st.info(f"Selected: {uploaded_file.name}")

    if process_button:

        try:
            with st.spinner("Processing your book..."):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    pdf_path = temp_file.name

                try:
                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    if not documents:
                        st.error("No readable pages found.")
                        st.stop()

                    # Split documents
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(documents)

                    if not chunks:
                        st.error("No text chunks were generated.")
                        st.stop()

                    # Create a unique Chroma collection
                    collection_name = (
                        "book_" + uuid.uuid4().hex[:20]
                    )

                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory="chroma-db",
                        collection_name=collection_name
                    )

                    # Create MMR retriever
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 5,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )

                    # Store in session
                    st.session_state.retriever = retriever
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.chat_history = []

                    st.success("Book processed successfully!")

                    st.write(f"Pages: {len(documents)}")
                    st.write(f"Chunks: {len(chunks)}")

                finally:
                    os.remove(pdf_path)

        except Exception as e:
            st.error(f"Document processing failed: {e}")

    st.divider()

    if st.session_state.book_name:

        st.success("Knowledge base ready")

        st.write(
            f"📄 {st.session_state.book_name}"
        )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------
# 6. MAIN CHAT INTERFACE
# --------------------------------------------------

if st.session_state.retriever is None:

    st.info(
        "👈 Upload a PDF book and click "
        "'Process Document' to get started."
    )

    st.markdown(
        """
        ### How it works

        1. Upload your PDF book.
        2. Extract text from the PDF.
        3. Split the text into chunks.
        4. Generate Gemini embeddings.
        5. Store embeddings in ChromaDB.
        6. Ask questions and receive answers using Groq.
        """
    )

else:

    st.subheader(
        f"💬 Chatting with: {st.session_state.book_name}"
    )

    # Display previous conversations
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # User question
    user_query = st.chat_input(
        "Ask anything about your book..."
    )

    if user_query:

        # Display user message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve relevant documents
        with st.chat_message("assistant"):

            with st.spinner("Searching your book..."):

                try:
                    docs = st.session_state.retriever.invoke(
                        user_query
                    )

                    context = "\n\n".join(
                        [
                            f"Page {doc.metadata.get('page', 0) + 1}:\n"
                            f"{doc.page_content}"
                            for doc in docs
                        ]
                    )

                    # Generate final prompt
                    final_prompt = prompt.invoke(
                        {
                            "context": context,
                            "question": user_query
                        }
                    )

                    # Generate response
                    response = llm.invoke(final_prompt)

                    answer = response.content

                    st.markdown(answer)

                    # Display source pages
                    with st.expander("📚 View source pages"):

                        for doc in docs:

                            page = doc.metadata.get(
                                "page", 0
                            ) + 1

                            st.markdown(
                                f"**Page {page}**"
                            )

                            st.write(doc.page_content)

                    # Save assistant response
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(f"Error generating answer: {e}")
