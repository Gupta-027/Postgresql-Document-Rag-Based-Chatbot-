# RAG_Learning

A from-scratch **Retrieval-Augmented Generation RAG chatbot** built to understand how modern AI knowledge assistants work end to end.

This project ingests a large PDF document, converts it into searchable chunks, stores those chunks as embeddings inside a Qdrant vector database, and allows users to chat with the PDF using natural language.

The current implementation uses the **PostgreSQL 18.4 documentation PDF**, which is around **1,400 pages**, as the knowledge source.

---

## Project Overview

`RAG_Learning` is a local-first RAG chatbot designed as a practical learning project for understanding the complete RAG pipeline.

The application supports:

- PDF loading
- Text chunking with overlap
- Local embedding generation using Ollama
- Vector storage using Qdrant
- Similarity search
- Context-aware answer generation using Google Gemini
- Chat-style frontend using Streamlit
- Conversation memory
- Sample queries and live status indicators

The goal of this project is not only to build a chatbot, but also to learn how each part of a production-grade RAG system works internally.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini `gemini-flash-latest` |
| Embeddings | Ollama `nomic-embed-text` |
| Vector Database | Qdrant |
| Orchestration | LangChain |
| Containerization | Docker Compose |
| Language | Python |

---

## How It Works

```text
PDF Document
    ↓
Load and Extract Text
    ↓
Split Text into Chunks
    ↓
Generate Embeddings Locally using Ollama
    ↓
Store Vectors in Qdrant
    ↓
User Asks a Question
    ↓
Embed the User Query
    ↓
Retrieve Similar Chunks from Qdrant
    ↓
Send Context + Question to Gemini
    ↓
Generate Final Answer
