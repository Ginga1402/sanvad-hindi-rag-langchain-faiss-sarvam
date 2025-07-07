# संवाद : Your Smart Hindi RAG Chatbot 🤖💬

### 🌐 An Open-Source Hindi RAG System Powered by LangChain, FAISS, and Sarvam-m LLM for Indic Language Intelligence

---

## 📘 Project Description

**Sanvad (संवाद)** is a powerful, open-source **Hindi Retrieval-Augmented Generation (RAG)** system built using the cutting-edge **LangChain framework**, **FAISS vector store**, and **Sarvam-m**, a recently released multilingual LLM optimized for Indic languages. Designed to deliver accurate, context-aware answers from your custom Hindi documents, Sanvad combines the best of **language models and vector search** to enable document intelligence and multilingual Q&A at scale.

At its core, Sanvad:

- Accepts user queries in **Hindi**.
- Retrieves relevant document chunks using **FAISS**.
- Passes context to **Sarvam-m** LLM via **LangChain** for generating highly relevant answers.
- Offers a plug-and-play architecture suitable for local or cloud-based deployment.

> ✨ **Powered by Sarvam-m:** A transformer-based multilingual instruction-tuned LLM (3B parameters), built by [Sarvam AI](https://sarvam.ai), optimized for **multilingual understanding**, especially Indian languages like Hindi, Tamil, and Bengali. It supports over 20 Indic languages and is trained on diverse datasets with a focus on both factuality and instruction-following tasks. Check it out here: [Sarvam-m on HuggingFace](https://huggingface.co/sarvamai/sarvam-m)

---

## 📁 Project Structure

```
Hindi-RAG/
├── ingest.py           # PDF ingestion and vector store creation
├── hindi_rag.py        # Core RAG functionality and model loading
├── flask_app.py        # API server
├── streamlit_app.py    # Web interface
├── data/
│   └── pdfs/              # Directory for PDF documents
├── vector_store/          # FAISS vector store files

└── .gitignore

```

