import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pymupdf4llm
from langchain.schema import Document


# ────────────────────────────────────────────────────────
# 🔧 Step 1: Setup Torch environment
# ────────────────────────────────────────────────────────

print("=" * 100)
print("🔍 Checking Torch GPU availability...")
try:
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA Available: {cuda_available}")
    DEVICE = "cuda" if cuda_available else "cpu"
    print(f"🧠 Using torch version: {torch.__version__} | Device: {DEVICE}")
except Exception as e:
    print(f"❌ Error checking Torch CUDA availability: {e}")
    DEVICE = "cpu"
print("=" * 100)



# ────────────────────────────────────────────────────────
# 📚 Step 2: Load Vector Store and Embeddings
# ────────────────────────────────────────────────────────


print("📦 Loading HuggingFace Embedding Model...")
try:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_kwargs = {'device': DEVICE}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    print(f"✅ Embedding model '{model_name}' loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    raise

# ──────────────────────────────────────────────────────────────────────
# 🔧 Step 3: Convert raw output to LangChain-compatible Document format
# ──────────────────────────────────────────────────────────────────────



def convert_to_documents(output):
    
    print("📄 Converting extracted data to LangChain Document format...")
    docs = []
    try:
        for idx, item in enumerate(output):
            metadata = item.get("metadata", {})
            doc = Document(
                page_content=item.get("text", ""),
                metadata={
                    "source": metadata.get("file_path"),
                    "file_path": metadata.get("file_path"),
                    "page": metadata.get("page"),
                    "total_pages": metadata.get("page_count"),
                    "format": metadata.get("format"),
                    "title": metadata.get("title"),
                    "author": metadata.get("author"),
                    "subject": metadata.get("subject"),
                    "keywords": metadata.get("keywords"),
                    "creator": metadata.get("creator"),
                    "producer": metadata.get("producer"),
                    "creationDate": metadata.get("creationDate"),
                    "modDate": metadata.get("modDate"),
                    "trapped": metadata.get("trapped"),
                }
            )
            docs.append(doc)
            if idx < 3:
                print(f"✅ Preview Document {idx + 1}: {doc}")
        print(f"📚 Total documents created: {len(docs)}\n")
    except Exception as e:
        print(f"❌ Error during document conversion: {e}")
        raise
    return docs

# ────────────────────────────────────────────────────────
# 📚 Step 4: Read PDF and convert to markdown
# ────────────────────────────────────────────────────────



pdf_path = "Data/7-4-23-220.pdf"
print(f"📥 Reading and parsing PDF: {pdf_path}")
try:
    raw_output = pymupdf4llm.to_markdown(doc=pdf_path, page_chunks=True)
    print(f"✅ PDF successfully parsed into markdown. Total items: {len(raw_output)}\n")
except Exception as e:
    print(f"❌ Failed to read PDF: {e}")
    raise


docs = convert_to_documents(raw_output)


# ──────────────────────────────────────────────────────────────────────
# 🔧 Step 5: Split the documents
# ──────────────────────────────────────────────────────────────────────


print("✂️ Splitting documents into smaller chunks...")
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Text splitting complete. Chunks generated: {len(chunks)}")
    if chunks:
        print(f"📝 First chunk preview:\n{chunks[0]}\n")
    else:
        print("⚠️ No chunks generated.")
except Exception as e:
    print(f"❌ Error during text splitting: {e}")
    raise


# ────────────────────────────────────────────────────────
# 🧠 Step 4: Vector store creation with FAISS
# ────────────────────────────────────────────────────────


print("📊 Generating vector store using FAISS...")
vectorstore_path = "./vector_store"
try:
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(vectorstore_path)
    print(f"✅ Vector store saved at: {vectorstore_path}")
except Exception as e:
    print(f"❌ Failed to create/save vector store: {e}")
    raise

print("🏁 Ingest pipeline completed successfully.")
