import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import math

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "true").lower() == "true"
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", 0.8))
SAFE_INTENTS = [i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL,INQUIRY").split(",")]
TEST_EMAIL = os.getenv("TEST_EMAIL", "komalsiddharth814@gmail.com")

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --------------------------
# Helper: Call OpenAI
# --------------------------
def call_openai(system_prompt: str, user_prompt: str) -> dict:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

# --------------------------
# Helper: Post reply to Freshdesk
# --------------------------
def post_freshdesk_reply(ticket_id: int, body: str):
    url = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets/{ticket_id}/reply"
    resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body}, timeout=20)
    if resp.status_code not in [200, 201]:
        logging.error(f"❌ Failed to post reply: {resp.status_code}, {resp.text}")
    else:
        logging.info(f"✅ Auto-reply posted to ticket {ticket_id}")

def post_freshdesk_note(ticket_id: int, body: str, private=True):
    url = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2/tickets/{ticket_id}/notes"
    resp = requests.post(url, auth=(FRESHDESK_API_KEY, "X"), json={"body": body, "private": private}, timeout=20)
    if resp.status_code not in [200, 201]:
        logging.error(f"❌ Failed to post note: {resp.status_code}, {resp.text}")
    else:
        logging.info(f"✅ Private note added to ticket {ticket_id}")

# --------------------------
# Webhook Route
# --------------------------
@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    payload = await request.json()
    logging.info(f"📩 Incoming payload: {json.dumps(payload)}")

    ticket = payload.get("ticket") or payload
    ticket_id = ticket.get("id")
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")
    requester_email = ticket.get("requester", {}).get("email", "").lower()

    if not ticket_id:
        return {"ok": False, "error": "ticket id missing"}

    if requester_email != TEST_EMAIL.lower():
        logging.info(f"⏭️ Ignored ticket from {requester_email}")
        return {"ok": True, "skipped": True}

    logging.info(f"✅ Processing ticket {ticket_id} from {requester_email}")

    # --------------------------
    # Generate AI response
    # --------------------------
    system_prompt = "You are a customer support assistant. Always respond in English."
    user_prompt = f"Ticket subject: {subject}\nTicket body: {description}\nReturn a JSON with keys: intent, confidence (0-1), reply_draft."

    try:
        ai_resp = call_openai(system_prompt, user_prompt)
        ai_text = ai_resp["choices"][0]["message"]["content"].strip()
        logging.info(f"🤖 OpenAI response: {ai_text}")
        parsed = json.loads(ai_text)
    except Exception as e:
        logging.error(f"❌ OpenAI error: {e}")
        parsed = {"intent": "UNKNOWN", "confidence": 0.0, "reply_draft": f"Hi, thanks for your message. Our team will contact you shortly."}

    intent = parsed.get("intent", "UNKNOWN").upper()
    confidence = float(parsed.get("confidence", 0.0))
    reply_draft = parsed.get("reply_draft", "")

    # --------------------------
    # Post private note (AI draft)
    # --------------------------
    note_body = f"""
**AI Assist Draft**
Ticket ID: {ticket_id}
Intent: {intent}
Confidence: {confidence}
Reply Draft: {reply_draft}
"""
    post_freshdesk_note(ticket_id, note_body, private=True)

    # --------------------------
    # Auto-reply if conditions met
    # --------------------------
    auto_reply_ok = ENABLE_AUTO_REPLY and intent in SAFE_INTENTS and confidence >= AUTO_REPLY_CONFIDENCE
    if auto_reply_ok:
        post_freshdesk_reply(ticket_id, reply_draft)
        logging.info(f"✅ Auto-replied to ticket {ticket_id}")
    else:
        logging.info(f"ℹ️ Auto-reply skipped (intent={intent}, confidence={confidence})")

    return {"ok": True, "ticket_id": ticket_id, "auto_reply_sent": auto_reply_ok}
