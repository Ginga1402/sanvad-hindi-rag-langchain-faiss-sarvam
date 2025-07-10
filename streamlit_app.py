import streamlit as st
from streamlit_chat import message
import tempfile
import logging
import torch
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import pymupdf4llm

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

st.set_page_config(
    page_title="संवाद : Your Smart Hindi RAG Chatbot 🤖💬",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stTitle { font-size: 3rem !important;
               background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               padding-bottom: 2rem; }
    .stSubheader { font-size: 1.5rem !important;
                   color: #FF8C00 !important;
                   text-align: center; }
    .stSidebar { background-color: rgba(35, 35, 35, 0.7) !important; }
    .stButton>button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stTextInput>div>div>input {
        background-color: rgba(35, 35, 35, 0.7) !important;
        color: #FFFFFF !important;
        border: 1px solid orange;
    }
    </style>
""", unsafe_allow_html=True)

# Title
col1, col2, col3 = st.columns([1,6,1])
with col2:
    st.title("संवाद : Your Smart Hindi RAG Chatbot   🤖💬")
    st.markdown("<h3 style='text-align: center; color: orange;'>Leveraging RAG (Retrieval-Augmented Generation) for Contextual, Knowledge-Infused Hindi Conversational AI</h3>", unsafe_allow_html=True)

# Sidebar
def clear_chat_history():
    st.session_state.messages = []

with st.sidebar:
    st.markdown("# 📚 Upload Document")
    st.markdown("---")
    uploaded_pdf = st.file_uploader("📥 Upload Hindi PDF", type=["pdf"])
    st.button('🧹 Clear Chat History', on_click=clear_chat_history)
    with st.expander("ℹ️ Help"):
        st.markdown("""
        **How to use:**
        1. 📤 Upload a Hindi PDF document
        2. ⏳ Wait for processing
        3. 💭 Ask questions about the content
        """)

# Cache LLM
@st.cache_resource(show_spinner=False)
def load_hindi_llm():
    tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-m")
    model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-m", torch_dtype="auto", device_map="auto")
    return tokenizer, model

def generate_answer(prompt, tokenizer, model):
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output = tokenizer.decode(output_ids)
        if "</think>" in output:
            return output.split("</think>")[-1].strip("</s>").strip()
        return output.strip("</s>").strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "उत्तर उत्पन्न करने में त्रुटि हुई।"

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        path = tmp_file.name

    st.info("📖 Processing uploaded PDF...")
    try:
        data = pymupdf4llm.to_markdown(doc=path, page_chunks=True)
        sample = " ".join([item['text'] for item in data])[:500]
        lang = detect(sample)

        if lang != "hi":
            st.error("❌ Please upload a Hindi language PDF only.")
            st.stop()

        docs = []
        for item in data:
            meta = item['metadata']
            docs.append({"page_content": item['text'], "metadata": meta})

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.create_documents([doc['page_content'] for doc in docs])

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        vector_store = FAISS.from_documents(split_docs, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        tokenizer, model = load_hindi_llm()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="🧑" if m["role"]=="user" else "🤖"):
                st.write(m["content"])

        if user_input := st.chat_input("अपना प्रश्न यहाँ टाइप करें..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🧑"):
                st.write(user_input)

            docs = retriever.invoke(user_input)
            context = " ".join([doc.page_content for doc in docs])
            prompt = f"""आप एक सहायक हैं। संदर्भ से उत्तर दीजिए:
                    संदर्भ: {context}
                    प्रश्न: {user_input}"""

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("सोच रहा हूँ..."):
                    output = generate_answer(prompt, tokenizer, model)
                    st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})

    except Exception as e:
        logging.error(f"PDF processing failed: {e}")
        st.error("❌ Document processing failed.")
else:
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h3 style='color: pink;'>A Hindi RAG Streamlit Chatbot leveraging Faiss Vector Store and SBERT embeddings for efficient, context-aware responses. Powered by Sarvam AI's multilingual LLM, it delivers intelligent, seamless conversations in Hindi.</h3>
        <p style='color: orange;'>कृपया शुरू करने के लिए एक हिंदी PDF अपलोड करें।</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p style='background: linear-gradient(45deg, orange, pink);
                  -webkit-background-clip: text;
                  -webkit-text-fill-color: transparent;'>
            Built with 💫 | संवाद v1.0
        </p>
    </div>
""", unsafe_allow_html=True)

