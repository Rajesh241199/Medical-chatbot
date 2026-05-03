# Medical Chatbot

A Retrieval-Augmented Generation (RAG) based medical chatbot that answers user questions from an indexed medical PDF knowledge base.

The application uses **Pinecone** for vector search, **HuggingFace embeddings** for document representation, **Ollama** for local LLM inference, and **Flask** for the web interface.

---

## Features

- Medical question-answering from a PDF knowledge base
- RAG pipeline using Pinecone vector search
- Local LLM inference with Ollama
- No OpenAI API credit required
- HuggingFace sentence-transformer embeddings
- Flask-based interactive chatbot UI
- Modular project structure
- Secure environment variable handling using `.env`

---

## Tech Stack

| Area | Tools |
|---|---|
| Backend | Python, Flask |
| LLM | Ollama, Llama 3.2 3B |
| Vector Database | Pinecone |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| RAG Framework | LangChain |
| Frontend | HTML, CSS, JavaScript |
| Environment | Anaconda / Python 3.10 |

---

## Project Structure

```text
Medical_Chatbot
│
├── Data
│   └── .gitkeep
│
├── research
│   └── trials.ipynb
│
├── src
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
│
├── static
│   └── style.css
│
├── templates
│   └── chat.html
│
├── app.py
├── store_index.py
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
└── README.md

User Question
    ↓
Flask Chat UI
    ↓
Retriever
    ↓
Pinecone Vector Index
    ↓
Relevant Medical Context
    ↓
Ollama Local LLM
    ↓
Final Answer

