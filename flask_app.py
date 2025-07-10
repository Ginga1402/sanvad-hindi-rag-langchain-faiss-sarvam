import torch
import datetime
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from hindi_rag import retriever,generate_hindi_answer


# ─────────────────────────────────────────────
# 🔧 Configure Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger()

# ─────────────────────────────────────────────
# 🔧 Initialize Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# 📊 In-Memory Stats
# ─────────────────────────────────────────────
api_stats = {
    "total_requests": 0,
    "successful_responses": 0,
    "fallback_responses": 0,
    "start_time": datetime.datetime.utcnow()
}

# ─────────────────────────────────────────────
# 🚀 Torch Device Info
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Torch device selected: {DEVICE}")



# ─────────────────────────────────────────────
# 🩺 Heartbeat
# ─────────────────────────────────────────────
@app.route("/heartbeat", methods=["GET"])
def heartbeat():
    log.info("Heartbeat check called.")
    return jsonify({
        "status": "alive",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }), 200

# ─────────────────────────────────────────────
# 📈 Stats Endpoint
# ─────────────────────────────────────────────
@app.route("/stats", methods=["GET"])
def stats():
    uptime = datetime.datetime.utcnow() - api_stats["start_time"]
    return jsonify({
        "total_requests": api_stats["total_requests"],
        "successful_responses": api_stats["successful_responses"],
        "fallback_responses": api_stats["fallback_responses"],
        "uptime_seconds": int(uptime.total_seconds())
    }), 200

# ─────────────────────────────────────────────
# 💬 Query Endpoint
# ─────────────────────────────────────────────
@app.route("/query", methods=["POST"])
def query():
    if request.method != "POST":
        log.warning("❌ Invalid HTTP method used.")
        return jsonify({"status": "error", "message": "Only POST method is allowed."}), 405

    api_stats["total_requests"] += 1
    start_time = datetime.datetime.now()

    try:
        data = request.json
        question = data.get("question", "").strip()

        if not question:
            log.warning("⚠️ Missing question in request.")
            return jsonify({"status": "error", "message": "Missing 'question' field."}), 400

        print(f"\n📥 Question: {question}")
        log.info(f"Received question: {question}")

        docs = retriever.invoke(question)
        if not docs:
            log.warning("No relevant documents found.")
            return jsonify({
                "status": "success",
                "answer": "माफ़ कीजिए, मुझे कोई प्रासंगिक जानकारी नहीं मिली।",
                "reasoning": ""
            }), 200

        context = " ".join([doc.page_content for doc in docs])
        reasoning, answer = generate_hindi_answer(question, context)

        # Hindi language check
        if not any(char in answer for char in "अआइईउऊऋएऐओऔकखगघ"):
            api_stats["fallback_responses"] += 1
            log.warning("Answer is not in Hindi. Fallback applied.")
            answer = "उत्तर उत्पन्न करने में त्रुटि हुई।"

        api_stats["successful_responses"] += 1
        response_time = round((datetime.datetime.now() - start_time).total_seconds(), 2)

        log.info(f"Answer generated in {response_time} seconds.")
        print(f"🗣️ Answer: {answer}\n")

        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "response_time_sec": response_time
        }), 200

    except Exception as e:
        log.error(f"❌ Internal error: {e}")
        return jsonify({"status": "error", "message": "🚨 Internal server error"}), 500

# ─────────────────────────────────────────────
# ▶️ Start Server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🚀 Starting Hindi-RAG Flask API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

