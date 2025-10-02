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

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "false").lower() == "true"
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", "0.95"))
SAFE_INTENTS = [i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL").split(",")]
TEST_EMAIL = "komalsiddharth814@gmail.com".lower()  # Only this email is processed

PAYMENT_AGENT_ID = int(os.getenv("PAYMENT_AGENT_ID", "0"))  # Agent ID for payment issues
PAYMENT_AGENT_EMAIL = os.getenv("PAYMENT_AGENT_EMAIL", "wathorerahul@yahoo.com")  # Agent email for logging/note

KNOWLEDGE_BASE_CSV = os.getenv("KNOWLEDGE_BASE_CSV", "courses.csv")  # Default to courses.csv
KNOWLEDGE_BASE_PDF = os.getenv("KNOWLEDGE_BASE_PDF", "faq.pdf")  # Optional PDF

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
    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ OpenAI API request failed: {e}")
        raise

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
            return text[:4000]
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        return ""

def extract_from_csv(file_path: str, query: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        df = pd.read_csv(file_path)
        query_words = query.lower().split()
        matches = df[df.apply(lambda row: any(word in str(val).lower() for val in row for word in query_words), axis=1)]
        if matches.empty:
            compulsory_keywords = ["fees", "certificate", "links", "course"]
            if any(kw in query.lower() for kw in compulsory_keywords):
                return df.to_string(index=False)[:4000]
        return matches.to_string(index=False)[:4000]
    except Exception as e:
        logging.error(f"CSV extraction error: {e}")
        return ""

def get_freshdesk_ticket(ticket_id: int) -> dict | None:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}?include=requester"
    try:
        resp = requests.get(url, auth=(FRESHDESK_API_KEY, "X"), timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logging.error("❌ Failed to fetch ticket %s: %s", ticket_id, e)
        return None

def get_master_ticket_id(ticket_id: int, ticket: dict = None) -> int:
    if not ticket:
        ticket = get_freshdesk_ticket(ticket_id)
    if not ticket:
        return ticket_id
    parent_id = ticket.get("merged_ticket_id") or ticket.get("custom_fields", {}).get("cf_parent_ticket_id")
    if parent_id:
        logging.info("🔀 Ticket %s merged into %s", ticket_id, parent_id)
        return parent_id
    return ticket_id

def update_freshdesk_ticket(ticket_id: int, updates: dict) -> bool:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    try:
        resp = requests.put(url, auth=(FRESHDESK_API_KEY, "X"), json=updates, timeout=20)
        resp.raise_for_status()
        logging.info("✅ Updated ticket %s with: %s", ticket_id, updates)
        return True
    except requests.exceptions.RequestException as e:
        logging.error("❌ Failed to update ticket %s: %s", ticket_id, e)
        return False

def post_freshdesk_note(ticket_id: int, body: str, private: bool = True) -> dict:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    try:
        resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body, "private": private}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"❌ Failed posting note to ticket {ticket_id}: {e}")
        raise

def post_freshdesk_reply(ticket_id: int, body: str) -> dict:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/reply"
    try:
        resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"❌ Failed posting reply to ticket {ticket_id}: {e}")
        raise

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
    except json.JSONDecodeError as e:
        logging.exception("❌ Failed to parse JSON payload: %s", e)
        return {"ok": False, "error": "invalid JSON"}

    ticket = payload.get("ticket") or payload
    ticket_id = ticket.get("id") or payload.get("id") or ticket.get("ticket_id") or payload.get("ticket_id")

    if not ticket_id:
        logging.error("❌ Ticket id missing in payload")
        return {"ok": False, "error": "ticket id not found"}

    full_ticket = get_freshdesk_ticket(ticket_id)
    if not full_ticket:
        logging.error("❌ Failed to fetch full ticket details for %s", ticket_id)
        return {"ok": False, "error": "failed to fetch ticket"}

    requester_email = full_ticket.get("requester", {}).get("email", "").lower()
    requester_name = full_ticket.get("requester", {}).get("name", "Customer")
    subject = full_ticket.get("subject", "")
    description = full_ticket.get("description", "")

    if not requester_email:
        logging.warning("⚠️ Requester email missing, skipping processing")
        return {"ok": True, "skipped": True, "reason": "missing requester_email"}

    if requester_email != TEST_EMAIL:
        logging.info("⏭️ Ignored ticket %s from %s (testing phase)", ticket_id, requester_email)
        return {"ok": True, "skipped": True, "reason": "non-test email"}

    master_id = get_master_ticket_id(ticket_id, full_ticket)

    query_terms = f"{subject} {description}"
    kb_content = ""
    if KNOWLEDGE_BASE_PDF:
        kb_content += "\nPDF Knowledge Base:\n" + extract_from_pdf(KNOWLEDGE_BASE_PDF, query_terms)

    if KNOWLEDGE_BASE_CSV:
        kb_content += "\nCSV Knowledge Base:\n" + extract_from_csv(KNOWLEDGE_BASE_CSV, query_terms)

    if kb_content:
        logging.info("📚 Extracted KB content length: %d", len(kb_content))
    else:
        logging.warning("⚠️ No KB content extracted; ensure files exist and are accessible.")

    # ---- FIXED SYSTEM PROMPT ----
    system_prompt = f"""
You are a professional customer support assistant for Team IMK. Always respond in English only.

STRICT RULES for reply_draft formatting:
- Output reply_draft as an HTML-formatted string for proper rendering in email systems like Freshdesk.
  Use <p> for paragraphs, <br> for line breaks, <ul><li> for bullet points, and <strong> for bold text.
- Keep tone polite, professional, and helpful at all times.
- Use short paragraphs (2–3 lines max) for readability; use <br> for line breaks where needed.
- For course-related queries:
  - Present details clearly using HTML bullet points (<ul><li>Course Name: ...</li></ul> etc.).
  - Include all relevant fields from the Knowledge Base (Fee, Enrollment Link, Certificate, Duration, Access, Other notes).
  - Never invent or assume missing details.
- For general queries (complaints, feedback, support requests):
  - Use structured HTML paragraphs (<p>...</p>) and bullet points only where they improve clarity.
- Always end emails with a warm closing in HTML:
  <p>Thanks & Regards,<br>Rahul<br>Team IMK<br>
  <img src="https://indattachment.freshdesk.com/inline/attachment?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA2MDAxNTMxMTAxOCwiZG9tYWluIjoibWl0ZXNoa2hhdHJpdHJhaW5pbmdsbHAuZnJlc2hkZXNrLmNvbSIsImFjY291bnRfaWQiOjMyMzYxMDh9.gswpN0f7FL4QfimJMQnCAKRj2APFqkOfYHafT0zB8J8" alt="Team IMK Logo" width="200" height="auto" style="display: block; margin: 0 auto;" /></p>
- Always use this HTML format for hyperlinks: <a href="https://example.com">Click Here</a>.
- Never merge multiple pieces of information into one block; enforce structure using HTML tags.
- Fallback: If the query cannot be answered from the Knowledge Base, politely acknowledge and suggest contacting support for further help.

OUTPUT REQUIREMENTS (JSON only):
- intent (one word)
- confidence (0–1)
- summary (2–3 lines summarizing user query)
- sentiment (Angry/Neutral/Positive)
- reply_draft (string: well-formatted, polite HTML email)
- kb_suggestions (list of short titles or URLs)

COURSE-RELATED TEMPLATE (HTML):
<p>Hi {requester_name},</p>
<p>Thank you for reaching out. My name is Rahul from <strong>Team IMK</strong>, and I’ll be assisting you today. Please find the course details below:</p>
<ul>
<li>Course Name: <Course Name></li>
<li>Course Fee: ₹<Fee></li>
<li>Enrollment Link: <a href="<Link>">Click here to Enroll</a></li>
<li>Certificate Provided: <Yes/No></li>
<li>Access: <Lifetime/Other></li>
<li>Duration: <Duration></li>
</ul>
<p>If you have further questions, feel free to ask.</p>
<p>Thanks & Regards,<br>Rahul<br>Team IMK<br>
<img src="https://indattachment.freshdesk.com/inline/attachment?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA2MDAxNTMxMTAxOCwiZG9tYWluIjoibWl0ZXNoa2hhdHJpdHJhaW5pbmdsbHAuZnJlc2hkZXNrLmNvbSIsImFjY291bnR_aWQiOjMyMzYxMDh9.gswpN0f7FL4QfimJMQnCAKRj2APFqkOfYHafT0zB8J8" alt="Team IMK Logo" width="200" height="auto" style="display: block; margin: 0 auto;" /></p>

GENERAL QUERY TEMPLATE (HTML):
<p>Hi {requester_name},</p>
<p>Thank you for reaching out. My name is Rahul from <strong>Team IMK</strong>, and I’ll be assisting you today.</p>
<p>[Insert professional AI reply here: use short, clear paragraphs and <ul><li> bullets where appropriate.]</p>
<p>If you have further questions, feel free to ask.</p>
<p>Thanks & Regards,<br>Rahul<br>Team IMK<br>
<img src="https://indattachment.freshdesk.com/inline/attachment?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA2MDAxNTMxMTAxOCwiZG9tYWluIjoibWl0ZXNoa2hhdHJpdHJhaW5pbmdsbHAuZnJlc2hkZXNrLmNvbSIsImFjY291bnR_aWQiOjMyMzYxMDh9.gswpN0f7FL4QfimJMQnCAKRj2APFqkOfYHafT0zB8J8" alt="Team IMK Logo" width="200" height="auto" style="display: block; margin: 0 auto;" /></p>
"""

    user_prompt = f"Customer: {requester_name}\nSubject: {subject}\nBody: {description}\n\nKnowledge Base:\n{kb_content}\n\nReturn valid JSON only."

    try:
        ai_resp = call_openai(system_prompt, user_prompt)
        assistant_text = ai_resp["choices"][0]["message"]["content"].strip()
        parsed = json.loads(assistant_text)
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logging.exception("⚠️ OpenAI or JSON parse error: %s", e)
        parsed = {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "summary": description[:200],
            "sentiment": "UNKNOWN",
            "reply_draft": f"<p>Hi {requester_name},</p><p>Thank you for your inquiry. Our support team will get back to you soon.</p><p>Thanks & Regards,<br>Rahul<br>Team IMK</p>",
            "kb_suggestions": []
        }

    intent = parsed.get("intent", "UNKNOWN").upper()
    confidence = parsed.get("confidence", 0.0)
    is_payment_issue = "PAYMENT" in intent or "BILLING" in intent or "REFUND" in intent

    # Handle payment issues: assign high priority and agent
    assignment_info = ""
    if is_payment_issue and PAYMENT_AGENT_ID > 0:
        updates = {
            "priority": 3,  # High priority in Freshdesk
            "assignee_id": PAYMENT_AGENT_ID
        }
        if update_freshdesk_ticket(master_id, updates):
            assignment_info = f"<p><strong>Assigned to:</strong> {PAYMENT_AGENT_EMAIL} (ID: {PAYMENT_AGENT_ID})</p><p><strong>Priority:</strong> High</p>"

    # Post private draft note with only the draft message reply displayed, keep buttons
    # Build special AI_DRAFT private note (only for app to pickup)
ai_draft_content = parsed.get("reply_draft", f"<p>Hi {requester_name},</p><p>Thank you for your inquiry. Our support team will get back to you soon.</p><p>Thanks & Regards,<br>Rahul<br>Team IMK</p>")

# Special format: Start with #AI_DRAFT, then pure draft, then internal info
note = f"""#AI_DRAFT

{ai_draft_content}

[Internal: AI Intent - {intent}, Confidence - {confidence:.2f}, Sentiment - {parsed.get('sentiment', 'Neutral')}]
{assignment_info}

**Summary:**
{parsed.get('summary', 'No summary available')}

**KB Suggestions:**
{json.dumps(parsed.get('kb_suggestions', []), ensure_ascii=False)}

<div style="margin-top: 20px;">
<a href="https://{FRESHDESK_DOMAIN}/a/tickets/{master_id}" style="background-color: #2196F3; color: white; padding: 10px 20px; margin-right: 10px; text-decoration: none; border-radius: 5px;">Edit</a>
<a href="https://{FRESHDESK_DOMAIN}/a/tickets/{master_id}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Share/Send</a>
</div>
{"⚠️ Payment-related issue → private draft only." if is_payment_issue else "_Note: AI draft — please review before sending._"}
"""
    try:
        post_freshdesk_note(master_id, note, private=True)
        logging.info("✅ Posted #AI_DRAFT private note to ticket %s", master_id)
    except Exception as e:
        logging.exception("❌ Failed posting note: %s", e)

    # No auto-reply sending during initial testing phase
    auto_reply_ok = False

    return {
        "ok": True,
        "ticket": ticket_id,
        "master_ticket": master_id,
        "intent": intent,
        "confidence": confidence,
        "requester_email": requester_email,
        "auto_reply": auto_reply_ok
    }

