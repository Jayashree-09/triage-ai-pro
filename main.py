import os
import csv

# ---------- Silence Logs ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load Model ----------
model = SentenceTransformer('all-MiniLM-L6-v2')


# ---------- Load Documents ----------
def load_docs(folder="data"):
    docs = []
    filenames = []

    if not os.path.exists(folder):
        os.makedirs(folder)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if not file.endswith(".txt"):
            continue

        with open(path, "r", encoding="utf-8") as f:
            docs.append(f.read())
            filenames.append(file)

    return docs, filenames


# ---------- Build Vector Store ----------
def build_vector_store(docs):
    return model.encode(docs)


# ---------- 🔥 RETRIEVE (WITH RE-RANK + FEEDBACK BOOST) ----------
def retrieve_answer(query, vectors, docs, feedback_db=None, top_k=3):

    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, vectors)[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        text = docs[idx][:300]
        score = float(scores[idx])

        # 🔥 FEEDBACK BOOSTING
        if feedback_db:
            for fb in feedback_db:
                if fb["text"] in text:
                    if fb["label"] == "up":
                        score += 0.1   # boost good answers
                    elif fb["label"] == "down":
                        score -= 0.1   # penalize bad answers

        results.append((text, score))

    # ---------- RE-RANK ----------
    query_words = set(query.lower().split())

    reranked = []
    for text, score in results:
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)

        new_score = score + (0.05 * overlap)
        reranked.append((text, new_score))

    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked


# ---------- MEMORY ----------
def get_memory_context(history, limit=3):
    if not history:
        return ""

    recent = history[-limit:]
    context = ""

    for item in recent:
        context += f"User: {item['user']}\n"
        context += f"Bot: {item['response']}\n"

    return context


# ---------- DOMAIN ----------
def detect_domain(text):
    text = text.lower()
    if "coding" in text or "hackerrank" in text:
        return "technical"
    elif "card" in text or "payment" in text:
        return "finance"
    elif "ai" in text:
        return "ai"
    return "general"


# ---------- INTENT ----------
def classify_intent(text):
    text = text.lower()
    if "refund" in text or "charged" in text:
        return "billing"
    elif "error" in text or "bug" in text:
        return "technical"
    elif "login" in text:
        return "account"
    elif "fraud" in text:
        return "fraud"
    return "general"


# ---------- RISK ----------
def risk_score(text):
    high_risk_words = ["fraud", "unauthorized", "hacked", "stolen"]
    score = 0

    for word in high_risk_words:
        if word in text.lower():
            score += 3

    if "urgent" in text.lower():
        score += 2

    return score


# ---------- PRIORITY ----------
def get_priority(risk):
    if risk >= 3:
        return "HIGH 🔴"
    elif risk == 2:
        return "MEDIUM 🟡"
    return "LOW 🟢"


# ---------- DECISION ----------
def decide_action(intent, risk, confidence):
    if risk >= 3:
        return "escalate", "High-risk detected"

    if confidence < 0.3:
        return "clarify", "Low confidence"

    if intent == "fraud":
        return "escalate", "Fraud issue"

    return "respond", "Sufficient confidence"


# ---------- RESPONSE ----------
def generate_response(action, answer, reason, confidence):

    explanation = f"\n\n🧠 Reason: {reason}"

    if action == "escalate":
        return f"⚠️ Escalating to human support.{explanation}\nConfidence: {confidence:.2f}"

    if action == "clarify":
        return f"🤔 Need more details.{explanation}\nConfidence: {confidence:.2f}"

    return f"✅ Solution:\n{answer}{explanation}\nConfidence: {confidence:.2f}"


# ---------- 🔥 FEEDBACK STORAGE ----------
def save_feedback(text, label, file="feedback.csv"):
    with open(file, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text, label])


def load_feedback(file="feedback.csv"):
    if not os.path.exists(file):
        return []

    feedback = []
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            feedback.append({"text": row[0], "label": row[1]})

    return feedback


# ---------- 🚀 AGENT PIPELINE (FINAL) ----------
def agent_pipeline(user_input, vectors, docs, history=None):

    steps = {}

    # STEP 1: PLAN
    domain = detect_domain(user_input)
    intent = classify_intent(user_input)
    risk = risk_score(user_input)
    steps["plan"] = "Intent + Risk analyzed"

    # STEP 2: MEMORY
    memory_context = get_memory_context(history) if history else ""
    steps["memory"] = "Context used"

    # STEP 3: LOAD FEEDBACK
    feedback_db = load_feedback()

    # STEP 4: RETRIEVE
    query = user_input + "\n" + memory_context
    results = retrieve_answer(query, vectors, docs, feedback_db)
    best_answer, confidence = results[0]
    steps["retrieval"] = "Docs retrieved + reranked"

    # STEP 5: FALLBACK
    if confidence < 0.35:
        r1 = retrieve_answer(user_input, vectors, docs, feedback_db)
        a1, c1 = r1[0]

        short_query = " ".join(user_input.split()[:5])
        r2 = retrieve_answer(short_query, vectors, docs, feedback_db)
        a2, c2 = r2[0]

        best_answer, confidence = max(
            [(best_answer, confidence), (a1, c1), (a2, c2)],
            key=lambda x: x[1]
        )

        steps["fallback"] = "Retry used"
    else:
        steps["fallback"] = "Not needed"

    # STEP 6: DECIDE
    action, reason = decide_action(intent, risk, confidence)

    # STEP 7: ACT
    response = generate_response(action, best_answer, reason, confidence)

    # STEP 8: VERIFY
    steps["verify"] = "Checked confidence"

    return {
        "response": response,
        "confidence": confidence,
        "domain": domain,
        "intent": intent,
        "risk": risk,
        "action": action,
        "reason": reason,
        "results": results
    }


# ---------- CLI TEST ----------
def main():
    docs, _ = load_docs()

    if not docs:
        print("⚠️ Add .txt files inside 'data'")
        return

    vectors = build_vector_store(docs)
    history = []

    while True:
        ticket = input("\nEnter ticket (or exit): ")
        if ticket == "exit":
            break

        output = agent_pipeline(ticket, vectors, docs, history)

        print("\nResponse:", output["response"])
        print("Priority:", get_priority(output["risk"]))

        # 🔥 simulate feedback
        fb = input("Feedback (up/down): ")
        if fb in ["up", "down"]:
            save_feedback(output["response"], fb)

        history.append({
            "user": ticket,
            "response": output["response"]
        })


if __name__ == "__main__":
    main()