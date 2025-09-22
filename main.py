import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
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
TEST_EMAIL = "komalsiddharth814@gmail.com"  # Only this email is processed

if not (FRESHDESK_DOMAIN and FRESHDESK_API_KEY and OPENAI_API_KEY):
    logging.warning("❌ Missing required env vars: FRESHDESK_DOMAIN, FRESHDESK_API_KEY, OPENAI_API_KEY.")

# --------------------------
# Initialize COURSES_DF
# --------------------------
try:
    COURSES_DF = pd.read_csv("courses.csv")  # Adjust path if needed
    logging.info("✅ Loaded COURSES_DF with %d rows", len(COURSES_DF))
except Exception as e:
    logging.error("❌ Failed to load COURSES_DF: %s", e)
    COURSES_DF = pd.DataFrame(columns=["Course Name"])  # Fallback empty DataFrame

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
    Robust extraction of requester email from payload
    Handles different Freshdesk webhook structures
    """
    logging.info("🔍 Payload for email extraction: %s", json.dumps(payload, ensure_ascii=False))
    ticket = payload.get("ticket") or payload

    # Direct checks
    if "requester" in ticket and "email" in ticket["requester"]:
        return ticket["requester"]["email"].lower()
    if "contact" in ticket and "email" in ticket["contact"]:
        return ticket["contact"]["email"].lower()
    if "requester_email" in ticket:
        return ticket["requester_email"].lower()
    if "email" in ticket:
        return ticket["email"].lower()
    if "from" in ticket:
        return ticket["from"].lower()

    # Fallback for nested structures
    try:
        return payload["ticket"]["requester"]["email"].lower()
    except:
        return ""


def get_course_details(course_name: str) -> dict:
    """
    Fetch course details from COURSES_DF based on course_name
    """
    if not course_name or not isinstance(course_name, str):
        logging.warning("⚠️ Invalid or empty course_name: %s", course_name)
        return {"course_name": course_name, "details": "No matching course found"}

    try:
        # Use regex=False to treat course_name as a literal string
        matched = COURSES_DF[COURSES_DF['Course Name'].str.contains(course_name, case=False, na=False, regex=False)]
        if not matched.empty:
            return matched.iloc[0].to_dict()
        return {"course_name": course_name, "details": "No matching course found"}
    except Exception as e:
        logging.exception("❌ Error in get_course_details for %s: %s", course_name, e)
        return {"course_name": course_name, "details": f"Error: {str(e)}"}


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

    # Extract requester email robustly
    requester_email = extract_requester_email(payload)
    logging.info("🔹 Extracted ticket_id: %s, requester_email: %s", ticket_id, requester_email)

    if not ticket_id:
        logging.error("❌ Ticket id missing in payload")
        return {"ok": False, "error": "ticket id not found"}

    if not requester_email:
        logging.warning("⚠️ Requester email missing, skipping auto-reply")
        return {"ok": True, "skipped": True, "reason": "missing requester_email"}

    if requester_email.lower() != TEST_EMAIL.lower():
        logging.info("⏭️ Ignored ticket %s from %s", ticket_id, requester_email)
        return {"ok": True, "skipped": True}

    # Get master ticket ID
    try:
        master_id = get_master_ticket_id(ticket_id)
        logging.info("🔀 Master ticket id: %s", master_id)
    except Exception as e:
        logging.exception("❌ Failed to get master ticket id: %s", e)
        master_id = ticket_id

    # Extract possible course (placeholder logic; replace with your actual logic)
    possible_course = subject.split("about")[-1].strip() if "about" in subject.lower() else ""
    logging.info("🔍 Extracted possible_course: %s", possible_course)
    possible_course = possible_course or ""
    if not possible_course.strip():
        logging.warning("⚠️ No valid course name extracted from ticket %s", ticket_id)
        course_details = {"course_name": "", "details": "No course specified"}
    else:
        course_details = get_course_details(possible_course)

    # AI classification
    system_prompt = (
        "You are a customer support assistant. Always respond in English only. "
        "Return JSON with: intent (one word), confidence (0-1), summary (2-3 lines), "
        "sentiment (Angry/Neutral/Positive), reply_draft (polite email reply using template), "
        "kb_suggestions (list of short titles or URLs).\n"
        "Include course details if provided: {course_details}\n"
        "Reply template:\n"
        "Dear [CustomerName],\n\n"
        "[Helpful AI reply]\n\n"
        "Best regards,\nSupport Team"
    ).format(course_details=json.dumps(course_details))
    user_prompt = f"Ticket subject:\n{subject}\n\nTicket body:\n{description}\n\nReturn valid JSON only."

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
    is_payment_issue = intent in ["BILLING", "PAYMENT"]

    # Build draft note
    note = f"""**🤖 AI Assist (draft)**

**Intent:** {intent}
**Confidence:** {confidence}
**Sentiment:** {parsed.get('sentiment')}
**Summary:** {parsed.get('summary')}
**Course Details:** {json.dumps(course_details, ensure_ascii=False)}
**Draft Reply:** {parsed.get('reply_draft')}
**KB Suggestions:** {json.dumps(parsed.get('kb_suggestions', []), ensure_ascii=False)}
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
        "course_details": course_details
    }
