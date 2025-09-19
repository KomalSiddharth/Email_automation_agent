import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Load .env locally
load_dotenv()

# --------------------------
# App + Logging Setup
# --------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------
# Env Variables
# --------------------------
FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "false").lower() == "true"
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", "0.95"))
SAFE_INTENTS = [i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL").split(",")]
TEST_EMAIL = "komlasiddharth814@gmail.com"  # ✅ Only this email is processed

if not (FRESHDESK_DOMAIN and FRESHDESK_API_KEY and OPENAI_API_KEY):
    logging.warning("❌ Missing required env vars: FRESHDESK_DOMAIN, FRESHDESK_API_KEY, OPENAI_API_KEY.")

# --------------------------
# Helpers
# --------------------------
def call_openai(system_prompt: str, user_prompt: str, max_tokens=600, temperature=0.0):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def get_freshdesk_ticket(ticket_id: int):
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

def post_freshdesk_note(ticket_id: int, body: str, private: bool = True):
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    auth = (FRESHDESK_API_KEY, "X")
    payload = {"body": body, "private": private}
    r = requests.post(url, auth=auth, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def post_freshdesk_reply(ticket_id: int, body: str):
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/reply"
    auth = (FRESHDESK_API_KEY, "X")
    payload = {"body": body}
    r = requests.post(url, auth=auth, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

# --------------------------
# Routes
# --------------------------
@app.get("/")
def root():
    return {"message": "AI Email Automation Backend Running"}

@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    payload = await request.json()
    logging.info("📩 Incoming Freshdesk payload: %s", payload)

    # Extract ticket details
    ticket_id = payload.get("id") or (payload.get("ticket") or {}).get("id")
    subject = payload.get("subject") or (payload.get("ticket") or {}).get("subject", "")
    description = payload.get("description") or (payload.get("ticket") or {}).get("description", "")
    requester_email = (
        payload.get("email")
        or (payload.get("ticket") or {}).get("email")
        or (payload.get("requester") or {}).get("email")
        or ""
    )

    if not ticket_id:
        logging.error("❌ Ticket id not found in payload")
        return {"ok": False, "error": "ticket id not found"}

    # ✅ Only process if test email
    if requester_email.lower() != TEST_EMAIL.lower():
        logging.info("⏭️ Ignored ticket %s from %s", ticket_id, requester_email)
        return {"ok": True, "skipped": True}

    # Check merged ticket → always post to master
    master_id = get_master_ticket_id(ticket_id)

    # AI classification
    system_prompt = (
        "You are a customer support assistant. Always respond in English only. "
        "Return JSON with: intent (one word), confidence (0-1), summary (2-3 lines), "
        "sentiment (Angry/Neutral/Positive), reply_draft (polite email reply using template), "
        "kb_suggestions (list of short titles or URLs).\n"
        "Reply template:\n"
        "Dear [CustomerName],\n\n"
        "[Helpful AI reply]\n\n"
        "Best regards,\nSupport Team"
    )
    user_prompt = f"""
Ticket subject:
{subject}

Ticket body:
{description}

Return valid JSON only.
"""

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

    # High priority if payment/billing
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
    if ENABLE_AUTO_REPLY and not is_payment_issue and intent in SAFE_INTENTS and confidence >= AUTO_REPLY_CONFIDENCE:
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
        "auto_reply": ENABLE_AUTO_REPLY and not is_payment_issue and intent in SAFE_INTENTS and confidence >= AUTO_REPLY_CONFIDENCE
    }
