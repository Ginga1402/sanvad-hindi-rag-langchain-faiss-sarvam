
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch



# ─────────────────────────────────────────────
# 🔧 Device Setup
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Torch device: {DEVICE}")

# ─────────────────────────────────────────────
# 📦 Load Hindi Model + Tokenizer
# ─────────────────────────────────────────────
model_name = "sarvamai/sarvam-m"
print(f"🔍 Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
print("✅ Hindi model and tokenizer loaded.\n")

# ─────────────────────────────────────────────
# 📚 Load Embeddings and Vector Store
# ─────────────────────────────────────────────
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_kwargs = {'device': DEVICE}
print(f"🔍 Loading embeddings: {embedding_model}")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs=embedding_kwargs)

vector_path = "/home/botadmin/Hindi-RAG/Hindi-RAG-V1/vector_store"
print(f"📂 Loading vector store from: {vector_path}")
vector_store = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3, "fetch_k": 6})
print("✅ Vector store ready.\n")

# ─────────────────────────────────────────────
# 🧾 Hindi Prompt Template
# ─────────────────────────────────────────────
prompt = PromptTemplate(
    template="""
        आप एक सम्मानीय सहायक हैं। आपका काम नीचे दिए गए संदर्भ से प्रश्नों का उत्तर देना है। आप केवल हिंदी भाषा में उत्तर दे सकते हैं। धन्यवाद। 
        You are never ever going to generate response in English. You are always going to generate response in Hindi no matter what. 
        You also need to keep your answer short and to the point. 
        संदर्भ: {context}
        प्रश्न: {question}
    """,
    input_variables=["question", "context"],
)

# ─────────────────────────────────────────────
# 🧠 Answer Generator
# ─────────────────────────────────────────────
def generate_hindi_answer(question, context):
    try:
        formatted_prompt = prompt.format(question=question, context=context)
        messages = [{"role": "user", "content": formatted_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids)

        if "</think>" in output_text:
            reasoning = output_text.split("</think>")[0].strip()
            answer = output_text.split("</think>")[-1].strip("</s>").strip()
        else:
            reasoning = ""
            answer = output_text.strip("</s>").strip()

        return reasoning, answer

    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "", "उत्तर उत्पन्न करने में त्रुटि हुई।"

