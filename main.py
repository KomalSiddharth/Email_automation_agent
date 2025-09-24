import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import PyPDF2
import pandas as pd

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

KNOWLEDGE_BASE_CSV = os.getenv("KNOWLEDGE_BASE_CSV", "courses.csv")  # Default to courses.csv as per requirements
KNOWLEDGE_BASE_PDF = os.getenv("KNOWLEDGE_BASE_PDF","faq.pdf")  # Optional, e.g., "faqs.pdf"

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
def call_openai(system_prompt: str, user_prompt: str, max_tokens=600, temperature=0.0) -> dict:
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
    response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def extract_from_pdf(file_path: str, query: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            query_words = query.lower().split()
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if any(word in page_text.lower() for word in query_words):
                    text += page_text + "\n\n"
            return text[:4000]  # Limit to avoid token overflow
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        return ""

def extract_from_csv(file_path: str, query: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        df = pd.read_csv(file_path)
        query_words = query.lower().split()
        # Filter rows where any column contains any word from the query
        matches = df[df.apply(lambda row: any(word in str(val).lower() for val in row for word in query_words), axis=1)]
        if matches.empty:
            # If no matches but CSV is compulsory for certain topics, include full if keywords present
            compulsory_keywords = ["fees", "certificate", "links", "course"]
            if any(kw in query.lower() for kw in compulsory_keywords):
                return df.to_string(index=False)[:4000]
        return matches.to_string(index=False)[:4000]
    except Exception as e:
        logging.error(f"CSV extraction error: {e}")
        return ""

def get_freshdesk_ticket(ticket_id: int) -> dict | None:
    url = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets/{ticket_id}"
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
    url = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets/{ticket_id}/notes"
    resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body, "private": private}, timeout=20)
    resp.raise_for_status()
    return resp.json()

def post_freshdesk_reply(ticket_id: int, body: str) -> dict:
    url = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets/{ticket_id}/reply"
    resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body}, timeout=20)
    resp.raise_for_status()
    return resp.json()

def extract_requester_info(payload: dict) -> tuple:
    """
    Robust extraction of requester email and name from payload
    Handles different Freshdesk webhook structures
    """
    logging.info("🔍 Payload for requester extraction: %s", json.dumps(payload, ensure_ascii=False))
    ticket = payload.get("ticket") or payload

    # Direct checks for email and name
    email = None
    name = "Customer"  # Default

    if "requester" in ticket:
        email = ticket["requester"].get("email")
        name = ticket["requester"].get("name", name)
    elif "contact" in ticket:
        email = ticket["contact"].get("email")
        name = ticket["contact"].get("name", name)
    elif "requester_email" in ticket:
        email = ticket["requester_email"]
    elif "email" in ticket:
        email = ticket["email"]
    elif "from" in ticket:
        email = ticket["from"]

    # Fallback for nested structures
    if not email:
        try:
            email = payload["ticket"]["requester"]["email"]
            name = payload["ticket"]["requester"].get("name", name)
        except:
            pass

    if email:
        email = email.lower()
    return email, name

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
        logging.info("📩 Incoming Freshdesk payload: %s", payload)
    except Exception as e:
        logging.exception("❌ Failed to parse JSON payload: %s", e)
        return {"ok": False, "error": "invalid JSON"}

    # Extract ticket details safely
    ticket = payload.get("ticket") or payload
    ticket_id = ticket.get("id") or payload.get("id")
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")

    # Extract requester email and name robustly
    requester_email, requester_name = extract_requester_info(payload)
    logging.info("🔹 Extracted ticket_id: %s, requester_email: %s, requester_name: %s", ticket_id, requester_email, requester_name)

    if not ticket_id:
        logging.error("❌ Ticket id missing in payload")
        return {"ok": False, "error": "ticket id not found"}

    if not requester_email:
        logging.warning("⚠️ Requester email missing, skipping auto-reply")
        return {"ok": True, "skipped": True, "reason": "missing requester_email"}

    if requester_email != TEST_EMAIL.lower():
        logging.info("⏭️ Ignored ticket %s from %s (testing phase)", ticket_id, requester_email)
        return {"ok": True, "skipped": True}

    # Fetch full ticket details to ensure complete data
    full_ticket = get_freshdesk_ticket(ticket_id)
    if full_ticket:
        subject = full_ticket.get("subject", subject)
        description = full_ticket.get("description", description)
        requester_name = full_ticket.get("requester", {}).get("name", requester_name)

    # Get master ticket ID
    try:
        master_id = get_master_ticket_id(ticket_id)
        logging.info("🔀 Master ticket id: %s", master_id)
    except Exception as e:
        logging.exception("❌ Failed to get master ticket id: %s", e)
        master_id = ticket_id

    # Extract KB content compulsorily for fees, certificate, links, course
    query_terms = f"{subject} {description}"  # Use ticket content as query
    kb_content = ""
    if KNOWLEDGE_BASE_PDF:
        kb_content += "\nPDF Knowledge Base:\n" + extract_from_pdf(KNOWLEDGE_BASE_PDF, query_terms)
    if KNOWLEDGE_BASE_CSV:
        kb_content += "\nCSV Knowledge Base (fees, certificates, links, courses):\n" + extract_from_csv(KNOWLEDGE_BASE_CSV, query_terms)
    if kb_content:
        logging.info("📚 Extracted KB content length: %d", len(kb_content))
    else:
        logging.warning("⚠️ No KB content extracted; ensure files exist and are accessible.")

    # AI classification
    system_prompt = (
        "You are a customer support assistant. Always respond in English only. "
        "Strictly use the provided Knowledge Base Context for any specific details like fees, certificates, links, courses, or FAQs. "
        "Compulsorily reference the CSV for fees, certificate, links, and course names if relevant. "
        "Do not invent or assume information not in the KB or your general knowledge. "
        "If the query cannot be answered from the KB, say so and suggest contacting support. "
        "Use common sense for general responses but prioritize KB accuracy. "
        "Return JSON with: intent (one word), confidence (0-1), summary (2-3 lines), "
        "sentiment (Angry/Neutral/Positive), reply_draft (polite email reply using template), "
        "kb_suggestions (list of short titles or URLs).\n"
        "Reply template:\n"
        "Dear {requester_name},\n\n"
        "[Helpful AI reply based strictly on KB]\n\n"
        "Best regards,\nSupport Team"
    ).format(requester_name=requester_name)  # Inject name into template
    user_prompt = f"Customer Name: {requester_name}\nTicket subject:\n{subject}\n\nTicket body:\n{description}\n\n"
    if kb_content:
        user_prompt += f"Knowledge Base Context:\n{kb_content}\n\n"
    user_prompt += "Return valid JSON only."

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
            "reply_draft": "AI parsing failed.",
            "kb_suggestions": []
        }

    intent = parsed.get("intent", "UNKNOWN").upper()
    confidence = parsed.get("confidence", 0.0)
    is_payment_issue = "PAYMENT" in intent or "BILLING" in intent  # More flexible match

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
        "auto_reply": auto_reply_ok
    }

