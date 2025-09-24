import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import PyPDF2
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK data (Render needs this to be explicit)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"❌ NLTK download failed: {e}")

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "true").lower() == "true"
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", "0.95"))
SAFE_INTENTS = [i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL").split(",")]
TEST_EMAIL = "komalsiddharth814@gmail.com"  # Only this email is processed

KNOWLEDGE_BASE_CSV = os.getenv("courses.csv", "data/courses.csv")  # Default path for Render
KNOWLEDGE_BASE_PDF = os.getenv("faq.pdf", "data/faq.pdf")  # Default path for Render

if not (FRESHDESK_DOMAIN and FRESHDESK_API_KEY and OPENAI_API_KEY):
    logging.warning("❌ Missing required env vars: FRESHDESK_DOMAIN, FRESHDESK_API_KEY, OPENAI_API_KEY.")

# --------------------------
# App & Logging
# --------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------
# Helper Functions
# --------------------------
def call_openai(system_prompt: str, user_prompt: str, max_tokens=600, temperature=0.2) -> dict:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"❌ OpenAI API call failed: {e}")
        raise

def preprocess_query(query: str) -> list:
    """Preprocess query into keywords, removing stop words."""
    try:
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [t for t in tokens if t.isalnum() and t not in stop_words]
        logging.info(f"🔍 Preprocessed query keywords: {keywords}")
        return keywords
    except Exception as e:
        logging.error(f"❌ Query preprocessing failed: {e}")
        return query.lower().split()

def extract_from_pdf(file_path: str, query: str) -> str:
    """Extract relevant text from PDF using keyword matching."""
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"⚠️ PDF file not found: {file_path}")
        return ""
    try:
        keywords = preprocess_query(query)
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                # Check if any keyword appears in the page
                if any(re.search(rf'\b{k}\b', page_text.lower(), re.IGNORECASE) for k in keywords):
                    text += f"\n[Page {page_num}]:\n{page_text}\n"
            if text:
                logging.info(f"📄 Extracted {len(text)} chars from PDF for query: {query[:50]}...")
                return text[:4000]  # Limit to avoid token overflow
            return ""
    except Exception as e:
        logging.error(f"❌ PDF extraction error: {e}")
        return ""

def extract_from_csv(file_path: str, query: str) -> str:
    """Extract relevant rows from CSV using keyword matching."""
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"⚠️ CSV file not found: {file_path}")
        return ""
    try:
        keywords = preprocess_query(query)
        df = pd.read_csv(file_path)
        # Filter rows where any column contains any keyword
        matches = df[df.apply(lambda row: any(
            any(re.search(rf'\b{k}\b', str(val).lower(), re.IGNORECASE) for k in keywords)
            for val in row
        ), axis=1)]
        if not matches.empty:
            result = matches.to_string(index=False)
            logging.info(f"📊 Extracted {len(result)} chars from CSV for query: {query[:50]}...")
            return result[:4000]
        return ""
    except Exception as e:
        logging.error(f"❌ CSV extraction error: {e}")
        return ""

def get_freshdesk_ticket(ticket_id: int) -> dict | None:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    resp = requests.get(url, auth=(FRESHDESK_API_KEY, "X"), timeout=20)
    if resp.status_code != 200:
        logging.error("❌ Failed to fetch ticket %s: %s", ticket_id, resp.text)
        return None
    return resp.json()

def get_master_ticket_id(ticket_id: int) -> int:
    ticket = get_freshdesk_ticket(ticket_id)
    if not ticket:
        return ticket_id
    parent_id = ticket.get("merged_ticket_id") or ticket.get("custom_fields", {}).get("cf_parent_ticket_id")
    if parent_id:
        logging.info("🔀 Ticket %s merged into %s", ticket_id, parent_id)
        return parent_id
    return ticket_id

def post_freshdesk_note(ticket_id: int, body: str, private: bool = True) -> dict:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body, "private": private}, timeout=20)
    resp.raise_for_status()
    return resp.json()

def post_freshdesk_reply(ticket_id: int, body: str) -> dict:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/reply"
    resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body}, timeout=20)
    resp.raise_for_status()
    return resp.json()

def extract_requester_email(payload: dict) -> str:
    """
    Robust extraction of requester email from payload for both ticket creation and replies
    Handles onTicketCreate and onConversationCreate webhook structures
    """
    logging.info("🔍 Payload for email extraction: %s", json.dumps(payload, ensure_ascii=False))
    ticket = payload.get("ticket") or payload
    conversation = payload.get("conversation") or {}

    email_paths = [
        (ticket.get("requester", {}).get("email"), "ticket.requester.email"),
        (ticket.get("contact", {}).get("email"), "ticket.contact.email"),
        (ticket.get("requester_email"), "ticket.requester_email"),
        (ticket.get("email"), "ticket.email"),
        (ticket.get("from"), "ticket.from"),
        (payload.get("requester", {}).get("email"), "requester.email"),
        (conversation.get("user", {}).get("email"), "conversation.user.email"),
        (conversation.get("from_email"), "conversation.from_email"),
    ]

    for email, path in email_paths:
        if email and isinstance(email, str) and '@' in email:
            logging.info(f"✅ Email extracted from {path}: {email.lower()}")
            return email.lower().strip()

    ticket_id = ticket.get("id") or payload.get("id")
    if ticket_id:
        try:
            ticket_data = get_freshdesk_ticket(ticket_id)
            if ticket_data and ticket_data.get("requester", {}).get("email"):
                email = ticket_data["requester"]["email"].lower().strip()
                logging.info(f"✅ Email fetched via API: {email}")
                return email
        except Exception as e:
            logging.error(f"❌ API fallback failed for ticket {ticket_id}: {e}")

    logging.warning("⚠️ No valid email found in payload")
    return ""

# --------------------------
# Routes
# --------------------------
@app.get("/")
def root():
    return {"message": "AI Email Automation Backend Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    try:
        payload = await request.json()
        logging.info("📩 Incoming Freshdesk payload: %s", json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        logging.exception("❌ Failed to parse JSON payload: %s", e)
        return {"ok": False, "error": "invalid JSON"}

    event_type = payload.get("eventType", "").lower()
    is_ticket_create = event_type == "onticketcreate"
    is_conversation_create = event_type == "onconversationcreate"

    if not (is_ticket_create or is_conversation_create):
        logging.warning("⚠️ Unsupported event type: %s", event_type)
        return {"ok": True, "skipped": True, "reason": "unsupported event"}

    ticket = payload.get("ticket") or payload
    ticket_id = ticket.get("id") or payload.get("id") or payload.get("conversation", {}).get("ticket_id")
    subject = ticket.get("subject", "") or "Reply to Ticket"
    description = ticket.get("description", "") or payload.get("conversation", {}).get("body_text", "")

    requester_email = extract_requester_email(payload)
    logging.info("🔹 Extracted ticket_id: %s, requester_email: %s", ticket_id, requester_email)

    if not ticket_id:
        logging.error("❌ Ticket id missing in payload")
        return {"ok": False, "error": "ticket id not found"}

    if not requester_email:
        logging.warning("⚠️ Requester email missing, skipping processing")
        return {"ok": True, "skipped": True, "reason": "missing requester_email"}

    if requester_email.lower() != TEST_EMAIL.lower():
        logging.info("⏭️ Ignored ticket %s from %s (not test email)", ticket_id, requester_email)
        return {"ok": True, "skipped": True, "reason": "non-test email"}

    try:
        master_id = get_master_ticket_id(ticket_id)
        logging.info("🔀 Master ticket id: %s", master_id)
    except Exception as e:
        logging.exception("❌ Failed to get master ticket id: %s", e)
        master_id = ticket_id

    # Extract KB content with enhanced query
    query_terms = f"{subject} {description}"
    kb_content = ""
    if KNOWLEDGE_BASE_PDF:
        pdf_content = extract_from_pdf(KNOWLEDGE_BASE_PDF, query_terms)
        if pdf_content:
            kb_content += f"\n=== PDF Knowledge Base ===\n{pdf_content}\n"
    if KNOWLEDGE_BASE_CSV:
        csv_content = extract_from_csv(KNOWLEDGE_BASE_CSV, query_terms)
        if csv_content:
            kb_content += f"\n=== CSV Knowledge Base ===\n{csv_content}\n"
    if kb_content:
        logging.info("📚 Extracted KB content length: %d", len(kb_content))
    else:
        logging.warning("⚠️ No KB content extracted for query: %s", query_terms[:50])

    # AI classification with strict KB enforcement
    system_prompt = (
        "You are a customer support assistant. Respond in English only. "
        "You MUST use the provided Knowledge Base (KB) content to answer all specific details (e.g., fees, duration, certification). "
        "If the KB lacks relevant information, explicitly state: 'The knowledge base does not contain this information. Please contact support.' "
        "Do NOT invent or assume details not in the KB. "
        "Use common sense for general responses but prioritize KB accuracy. "
        "Return JSON with: intent (one word, e.g., COURSE_INQUIRY), confidence (0-1), summary (2-3 lines), "
        "sentiment (Angry/Neutral/Positive), reply_draft (polite email reply using template), "
        "kb_suggestions (list of matched KB sections or empty). "
        "Reply template:\n"
        "Dear [CustomerName],\n\n"
        "[Answer based strictly on KB, or state KB lacks info]\n\n"
        "Best regards,\nSupport Team"
    )
    user_prompt = (
        f"=== Ticket Details ===\n"
        f"Subject: {subject}\n"
        f"Body: {description}\n\n"
        f"=== Knowledge Base Context ===\n"
        f"{kb_content if kb_content else 'No relevant KB content found.'}\n\n"
        "Return valid JSON only."
    )

    try:
        ai_resp = call_openai(system_prompt, user_prompt)
        assistant_text = ai_resp["choices"][0]["message"]["content"].strip()
        logging.info("🤖 OpenAI raw response: %s", assistant_text)
        parsed = json.loads(assistant_text)
    except Exception as e:
        logging.exception("⚠️ OpenAI or JSON parse error: %s", e)
        parsed = {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "summary": description[:200],
            "sentiment": "UNKNOWN",
            "reply_draft": "Dear Customer,\n\nWe could not process your request due to an error. Please contact support.\n\nBest regards,\nSupport Team",
            "kb_suggestions": []
        }

    intent = parsed.get("intent", "UNKNOWN").upper()
    confidence = parsed.get("confidence", 0.0)
    is_payment_issue = intent in ["BILLING", "PAYMENT"]

    # Build draft note
    note = f"""**🤖 AI Assist (draft)**

**Intent:** {intent}
**Confidence:** {confidence}

**Sentiment:** {parsed.get('sentiment')}

**Summary:**
{parsed.get('summary')}

**Draft Reply:**
{parsed.get('reply_draft')}

**KB Suggestions:**
{json.dumps(parsed.get('kb_suggestions', []), ensure_ascii=False)}

{"⚠️ Payment-related issue → private draft only." if is_payment_issue else "_Note: AI draft — please review before sending._"}
"""
    try:
        post_freshdesk_note(master_id, note, private=True)
        logging.info("✅ Posted private draft to ticket %s", master_id)
    except Exception as e:
        logging.exception("❌ Failed posting note: %s", e)

    # Auto-reply if safe
    auto_reply_ok = ENABLE_AUTO_REPLY and not is_payment_issue and intent in SAFE_INTENTS and confidence >= AUTO_REPLY_CONFIDENCE
    if auto_reply_ok:
        try:
            post_freshdesk_reply(master_id, parsed.get("reply_draft", ""))
            logging.info("✅ Auto-replied to ticket %s", master_id)
        except Exception as e:
            logging.exception("❌ Failed posting auto-reply: %s", e)
    else:
        logging.info("ℹ️ Auto-reply skipped (intent/setting)")

    return {
        "ok": True,
        "ticket": ticket_id,
        "master_ticket": master_id,
        "intent": intent,
        "confidence": confidence,
        "requester_email": requester_email,
        "auto_reply": auto_reply_ok,
        "event_type": event_type
    }
