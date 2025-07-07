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


---

## 💡 Use Cases

- 📚 **Hindi Document QA**: Upload Hindi PDFs and ask natural-language questions.
- 🏛️ **Government or Legal Docs**: Parse and query vernacular policies and notices.
- 🎓 **Education**: Hindi academic content search and tutoring assistant.
- 📄 **Enterprise Knowledge Bases**: Chat with internal docs in Hindi.
- 🗣️ **Multilingual Bots**: Add Sarvam-m for Hindi/Indic capabilities in existing assistants.

---

## ⚙️ Installation Instructions

Follow these simple steps to get the Voice to Text functionality running on your local machine:

1. Clone the repository to your local machine:
    ```bash
   git clone https://github.com/Ginga1402/sanvad-hindi-rag-langchain-faiss-sarvam.git
    ```
2. Navigate into the project directory:
    ```bash
    cd sanvad-hindi-rag-langchain-faiss-sarvam
    ```
3. Set up a virtual environment (recommended for Python projects):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
    ```
4. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## 🚀 Usage


### 1. **Ingest Data**


```bash
python ingest.py
```

### 2. Run the Flask application

```bash
python flask_app.py
```

### 3. Run Streamlit Chat application


```bash
streamlit run streamlit_app.py
```

## 🧰 Technologies Used in संवाद (Sanvad)

Below is the comprehensive tech stack used in the development of **Sanvad**, your smart Hindi RAG chatbot. Each tool plays a crucial role in building a robust, multilingual, document-aware conversational AI.

| Technology | Description | Link |
|------------|-------------|------|
| **LangChain** | A framework to build applications using LLMs with modular components like retrievers, chains, memory, and agents. Enables RAG orchestration. | [LangChain](https://www.langchain.com/) |
| **FAISS** | Facebook AI Similarity Search - Efficient library for vector similarity search and clustering, used for document retrieval. | [FAISS](https://faiss.ai/) |
| **Sarvam-m LLM** | Multilingual LLM optimized for Indic languages (Hindi, Tamil, Bengali). Instruction-tuned and open-source from Sarvam AI. Central to Hindi Q&A. | [Sarvam-m on HuggingFace](https://huggingface.co/sarvamai/sarvam-m) |
| **Hugging Face Transformers** | Popular library for loading and using pre-trained transformer models like Sarvam-m. Offers API for generation and tokenization. | [Transformers](https://huggingface.co/docs/transformers/index) |
| **Python** | The primary programming language used to develop Sanvad’s backend, data processing, and model integration. | [Python](https://www.python.org/) |
| **Flask** | Lightweight Python web framework used to serve APIs or expose the chatbot backend. Ideal for scalable microservice deployments. | [Flask](https://flask.palletsprojects.com/) |
| **Streamlit** *(optional)* | Python-based rapid frontend framework to build interactive web UIs for ML and LLM-based applications. Great for quick prototyping. | [Streamlit](https://streamlit.io/) |
| **Tiktoken** *(optional)* | A fast tokenizer developed by OpenAI. Useful for counting tokens to manage prompt size when interacting with LLMs. | [Tiktoken](https://github.com/openai/tiktoken) |

---

> ✅ This tech stack enables **semantic search**, **Hindi-native LLM responses**, and **real-time retrieval**, making Sanvad a production-ready solution for Indic document intelligence and question answering.

