import os
import csv
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import pandas as pd
import smtplib
import streamlit as st
from openai import OpenAI

# ----------------------
# Session state
# ----------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ----------------------
# Config / constants
# ----------------------
recruiter_keywords = [
    "recruiter", "hiring", "hire", "talent", "hr", "human resources",
    "job", "role", "position", "opening", "vacancy", "application",
    "candidate", "interview", "shortlist", "cv", "resume",
]

SENDER_EMAIL = "rubinaalmas610@gmail.com"
RESUME_PATH = "Rubina_Almas_Resume.pdf"
LOG_PATH = "rubiverse_logs.csv"

# 👉 New: admin analytics key (comes from secrets, not hard-coded)
ADMIN_ANALYTICS_KEY = st.secrets.get("RUBIVERSE_ADMIN_KEY", "")

# ⚠️ Use your real key here
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------
# Knowledge loading
# ----------------------
def load_docs(path: str = "rubina_knowledge.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks


docs = load_docs()


def build_doc_embeddings(docs_list):
    if not docs_list:
        return np.array([])
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=docs_list,
    )
    vectors = np.array([d.embedding for d in response.data])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors


doc_vectors = build_doc_embeddings(docs)


def retrieve_relevant_chunks(query: str, k: int = 4):
    if doc_vectors.size == 0:
        return []
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    qvec = np.array(resp.data[0].embedding)
    qvec = qvec / np.linalg.norm(qvec)

    sims = np.dot(doc_vectors, qvec)
    topk_idx = sims.argsort()[::-1][:k]
    return [docs[i] for i in topk_idx]


# ----------------------
# Logging & recruiter mode
# ----------------------
def detect_recruiter_mode(text: str) -> bool:
    t = text.lower()
    return any(keyword in t for keyword in recruiter_keywords)


def log_interaction(
    question: str,
    answer: str,
    recruiter_mode: bool,
    email_requested: bool = False,
):
    file_exists = os.path.exists(LOG_PATH)
    answer_preview = answer[:300].replace("\n", " ")

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp_utc",
                    "question",
                    "answer_preview",
                    "recruiter_mode",
                    "email_requested",
                ]
            )
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                question,
                answer_preview,
                int(recruiter_mode),
                int(email_requested),
            ]
        )


# ----------------------
# Core Rubiverse answer
# ----------------------
def ask_rubiverse(question: str):
    recruiter_mode = detect_recruiter_mode(question)

    context_chunks = retrieve_relevant_chunks(question, k=4)
    context_text = (
        "\n\n".join(context_chunks) if context_chunks else "No context available."
    )

    system_prompt = (
        "You are Rubiverse 🌌, a confident and positive AI assistant that ONLY talks about Rubina Almas. "
        "Your primary goal is to help recruiters and professionals understand why Rubina is an excellent candidate for their role. "
        "Always highlight Rubina's real skills, strengths, leadership qualities, and unique personality. "
        "Use strong, professional language that shows Rubina as capable, proactive, and value-driven. "
        "Keep it conversational — do NOT label responses as 'Summary', 'Top strengths', or similar. "
        "If the user asks broadly about Rubina's 'previous experience', 'experience' , 'background', or 'what has she done', interpret this as a request for her job experience and summarize her key roles, companies, and responsibilities from the context. "
        "Stay honest — never invent new claims or exaggerate information not provided in the context. "
        "If a question is inappropriate, offensive, objectifying, or disrespectful toward Rubina, "
        "respond by saying it is not appropriate to speak about a person that way and kindly encourage respectful communication. "
        "If asked something outside the context or unknown, say 'I do not know that yet!' instead of guessing. "
        "If a recruiter or someone asks for deeper technical details, portfolio documents, or collaboration, politely direct them to contact Rubina at rubinaalmas610@gmail.com — while still giving a helpful brief response. "
        "Be encouraging, warm, and clear about why Rubina is a great fit for opportunities. "
    )

    if recruiter_mode:
        system_prompt += (
            "The current user appears to be a recruiter. "
            "Respond in a friendly, natural, and professional tone. "
            "Keep it conversational — do NOT label responses as 'Summary', 'Top strengths', or similar. "
            "Highlight Rubina's most relevant strengths and achievements for the kind of role they mention. "
            "If they mention a specific industry or company, adapt the examples accordingly. "
        )

    system_prompt += f"\n\nCONTEXT:\n{context_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, temperature=0.3
    )
    answer = response.choices[0].message.content.strip()
    return answer, recruiter_mode


# ----------------------
# Email sending
# ----------------------
def send_resume_email(
    recipient_email: str,
    recipient_name: str | None = None,
    company: str | None = None,
):
    app_password = st.secrets["GMAIL_APP_PASSWORD"]
  
    if not os.path.exists(RESUME_PATH):
        raise FileNotFoundError(f"Resume file not found at: {RESUME_PATH}")

    msg = EmailMessage()
    to_name = recipient_name if recipient_name else "there"
    subject_company = f" at {company}" if company else ""

    msg["Subject"] = f"Rubina Almas – Resume{subject_company}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email

    body_lines = [
        f"Hi {to_name},",
        "",
        "Thank you for your interest in Rubina Almas.",
        "Please find her resume attached to this email.",
        "",
        "If you'd like to know more about her projects, skills, or experience,",
        "you can also chat with Rubiverse (her AI assistant) or email her directly at:",
        "  📩  rubinaalmas610@gmail.com",
        "",
        "Best regards,",
        "Rubiverse – on behalf of Rubina",
    ]
    msg.set_content("\n".join(body_lines), subtype="plain", charset="utf-8")

    with open(RESUME_PATH, "rb") as f:
        resume_data = f.read()
    msg.add_attachment(
        resume_data,
        maintype="application",
        subtype="pdf",
        filename=os.path.basename(RESUME_PATH),
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, app_password)
        smtp.send_message(msg)


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Rubiverse 🌌", page_icon="✨")
st.title("🌌 Welcome to Rubiverse 🌌")
st.write(
    "Ask anything about **Rubina Almas** – skills, projects, experience, and more ✨"
)

st.markdown("---")
st.subheader("📨 For Recruiters: Request Rubina's Resume")

with st.form("resume_request_form"):
    rec_name = st.text_input("Your name (optional)")
    rec_company = st.text_input("Company (optional)")
    rec_email = st.text_input("Work email", placeholder="name@company.com")
    submit_resume = st.form_submit_button("Email me Rubina's resume")
    
    if submit_resume:
        if not rec_email.strip():
            st.error("Please enter a valid email address.")
        else:
            try:
                send_resume_email(
                    rec_email.strip(),
                    rec_name.strip() or None,
                    rec_company.strip() or None,
                )
                        
                st.success(
                    "✅ Rubina's resume has been emailed to you. Thank you for your interest!"
                )
                    
                try:
                    log_interaction(
                        question=f"Resume requested by {rec_name or 'unknown'} at {rec_company or 'unknown'} ({rec_email})",
                        answer="Resume sent via email.",
                        recruiter_mode=True,
                        email_requested=True,
                    )
                
                except Exception as log_err:
                    st.warning(f"Could not log this recruiter request: {log_err}")

            except Exception as e:
                st.error(f"Sorry, something went wrong while sending the email: {e}")


st.markdown("---")

# Only show analytics section if an admin key is configured in secrets
if ADMIN_ANALYTICS_KEY:
    with st.expander("📊 Rubiverse analytics (owner view)"):
        admin_key_input = st.text_input(
            "Enter admin key to view analytics", type="password"
        )

        if admin_key_input == ADMIN_ANALYTICS_KEY:
            if not os.path.exists(LOG_PATH):
                st.info("No interactions logged yet.")
            else:
                df = pd.read_csv(LOG_PATH)

                st.subheader("Overview")
                total = len(df)
                rec_count = int(df["recruiter_mode"].sum())
                normal = total - rec_count
                email_requests = int(df["email_requested"].sum())

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total interactions", total)
                with col2:
                    st.metric("Recruiter-style questions", rec_count)
                with col3:
                    st.metric("Resume email requests", email_requests)

                df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.date
                daily_counts = df.groupby("date").size().reset_index(name="count")

                st.subheader("Interactions per day")
                if not daily_counts.empty:
                    daily_counts = daily_counts.set_index("date")
                    st.line_chart(daily_counts)
                else:
                    st.write("No data yet.")

                st.subheader("Recent questions")
                st.dataframe(
                    df.sort_values("timestamp_utc", ascending=False)[
                        [
                            "timestamp_utc",
                            "question",
                            "answer_preview",
                            "recruiter_mode",
                            "email_requested",
                        ]
                    ].head(20)
                )
        elif admin_key_input:
            st.error("Wrong admin key.")

if not docs:
    st.error("No knowledge found. Please make sure 'rubina_knowledge.txt' exists.")
else:
    # 1️⃣ Show existing chat history as real chat bubbles
    for role, msg in st.session_state["chat_history"]:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg)

    # 2️⃣ Chat-style input at the bottom
    user_question = st.chat_input(
        "Ask about Rubina's skills, projects, or experience..."
    )

    # 3️⃣ Handle new message (fires immediately on first send)
    if user_question:
        question = user_question.strip()
        if not question:
            st.error("Please type a question first 🙂")
        else:
            # Show the user's message immediately
            st.session_state["chat_history"].append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            # Get Rubiverse's answer
            with st.chat_message("assistant"):
                with st.spinner("Consulting the Rubiverse..."):
                    answer, recruiter_mode = ask_rubiverse(question)
                    st.markdown(answer)

            # Save answer to history
            st.session_state["chat_history"].append(("assistant", answer))

            # Log interaction
            try:
                log_interaction(
                    question, answer, recruiter_mode, email_requested=False
                )
            except Exception as e:
                st.warning(f"Could not log this interaction: {e}")


