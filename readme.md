
# 📚 BookMind AI – RAG Book Chat Application

An AI-powered document question-answering application that allows users to upload PDF books and interact with their contents using Retrieval-Augmented Generation (RAG).

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini Embeddings
- ChromaDB
- Groq GPT-OSS-20B
- PyPDF

## Features

- Upload PDF books
- Extract and split document text
- Generate vector embeddings
- Store embeddings in ChromaDB
- Retrieve relevant chunks using MMR
- Ask questions about uploaded documents
- Generate context-grounded answers
- Display source pages

## Architecture

PDF Upload → Text Extraction → Chunking →
Gemini Embeddings → ChromaDB → MMR Retrieval →
Groq LLM → AI Response

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RAG-BookMind-AI.git
cd RAG-BookMind-AI
```

Install dependencies:

```bash
uv sync
```

Configure your `.env`:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

Run the application:

```bash
uv run streamlit run app.py
```

## Future Improvements

- Multiple PDF uploads
- Conversation memory
- Source citations
- Document management
- Streaming responses
