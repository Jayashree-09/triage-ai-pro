import uuid
import streamlit as st
import re
from main import *

# ---------- Page Config ----------
st.set_page_config(page_title="TriageAI Pro", layout="wide")

# ---------- UI Styling ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #e2e8f0;
}

div[data-testid="stMetric"] {
  
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
   
}

.stChatMessage {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🚀 TriageAI Pro")
st.caption("Next-Gen AI Support Ticket Intelligence System")

# ---------- Highlight Function ----------
def highlight(text, query):
    for word in query.split():
        text = re.sub(f"({word})", r"**\\1**", text, flags=re.IGNORECASE)
    return text

# ---------- Load Data ----------
@st.cache_data
def load_cached_docs():
    return load_docs()

docs, _ = load_cached_docs()

if not docs:
    st.warning("⚠️ No data found")
    st.stop()

@st.cache_resource
def get_vector_store(docs):
    return build_vector_store(docs)

vectors = get_vector_store(tuple(docs))  # tuple = hashable

# ---------- Session ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Layout ----------
col1, col2 = st.columns([3, 1])

# ================= CHAT =================
with col1:
    st.subheader("💬 AI Support Chat")

    user_input = st.chat_input("💬 Describe your issue...")

    # ---------- PROCESS INPUT ----------
    if user_input:
        ticket_id = str(uuid.uuid4())[:8]

        output = agent_pipeline(user_input, vectors, docs, st.session_state.history) 
        response = output["response"]
        confidence = output["confidence"]
        domain = output["domain"]
        intent = output["intent"]
        risk = output["risk"]
        action = output["action"]
        reason = output["reason"]
        results = output["results"]
        priority = get_priority(risk)

        status = "Escalated" if action == "escalate" else "Resolved"

        st.session_state.history.append({
            "ticket_id": ticket_id,
            "user": user_input,
            "response": response,
            "priority": priority,
            "status": status,
            "confidence": confidence,
            "domain": domain,
            "intent": intent,
            "reason": reason,
            "results": results
        })

    # ---------- DISPLAY CHAT ----------
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(f"""
            <div style='
               color:white;
               padding:9px;
               border-radius:10px;
               max-width:100%;
               word-wrap:break-word;
               overflow-wrap:break-word;
            '>
           🎫 <b>Ticket ID:</b> {item['ticket_id']}<br>
          {item["user"]}
          </div>
          """, unsafe_allow_html=True)

        with st.chat_message("assistant"):
            st.markdown(f"""
            <div style='
               color:white;
               padding:12px;
               border-radius:10px;
               max-width:100%;
               word-wrap:break-word;
               overflow-wrap:break-word;
            '>
            {item["response"]}<br><br>
            <b>Status:</b> {item['status']} | <b>Priority:</b> {item['priority']}
            </div>
            """, unsafe_allow_html=True)

    # ---------- TOP MATCHES ----------
    if user_input and st.session_state.history:
        st.divider()
        st.subheader("🔍 Top AI Matches")

        last = st.session_state.history[-1]

        for i, (ans, score) in enumerate(last["results"]):
            st.write(f"**{i+1}. Confidence: {score:.2f}**")
            st.write(highlight(ans, last["user"]))

# ================= SIDEBAR =================
with col2:
    st.markdown("### 📊 Dashboard Insights")

    total = len(st.session_state.history)
    resolved = sum(1 for t in st.session_state.history if t["status"] == "Resolved")
    escalated = sum(1 for t in st.session_state.history if t["status"] == "Escalated")

    st.metric("📨 Total Tickets", total)
    st.metric("✅ Resolved", resolved)
    st.metric("⚠️ Escalated", escalated)

    # ✅ FIXED division issue
    if total > 0:
        resolution_rate = (resolved / total) * 100
        st.metric("📈 Resolution Rate", f"{resolution_rate:.1f}%")

        avg_conf = sum(t["confidence"] for t in st.session_state.history) / total
        st.metric("📊 Avg Confidence", f"{avg_conf:.2f}")
    else:
        st.metric("📈 Resolution Rate", "0%")
        st.metric("📊 Avg Confidence", "0.00")

    # ---------- AI Reasoning ----------
    if st.session_state.history:
        st.divider()
        st.subheader("🧠 AI Reasoning")

        last = st.session_state.history[-1]

        st.write(f"**Domain:** {last['domain']}")
        st.write(f"**Intent:** {last['intent']}")
        st.write(f"**Priority:** {last['priority']}")
        st.write(f"**Confidence:** {last['confidence']:.2f}")
        st.write(f"**Decision:** {last['status']}")
        st.write(f"**Reason:** {last['reason']}")

        # ---------- Feedback ----------
        st.subheader("👍 Feedback")

        if st.button("👍 Helpful"):
            st.success("Feedback recorded")

        if st.button("👎 Not Helpful"):
            st.warning("We'll improve this")