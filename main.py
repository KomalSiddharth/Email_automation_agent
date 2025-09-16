# main.py
import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Local development: load .env if present
load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")  # e.g. yourcompany.freshdesk.com
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")  # change if you prefer other model
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

if not (FRESHDESK_DOMAIN and FRESHDESK_API_KEY and OPENAI_API_KEY):
    logging.warning("One or more required env vars are missing. Set FRESHDESK_DOMAIN, FRESHDESK_API_KEY, OPENAI_API_KEY.")

@app.get("/")
def root():
    return {"message": "AI Email Automation Backend Running"}

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

def post_freshdesk_note(ticket_id: int, body: str, private: bool = True):
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    auth = (FRESHDESK_API_KEY, "X")
    payload = {"body": body, "private": private}
    r = requests.post(url, auth=auth, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    payload = await request.json()
    logging.info("Webhook payload: %s", payload)

    # Extract fields sent from Freshdesk automation (matches your webhook body)
    ticket_id = payload.get("id") or (payload.get("ticket") or {}).get("id")
    subject = payload.get("subject") or (payload.get("ticket") or {}).get("subject","")
    description = payload.get("description") or (payload.get("ticket") or {}).get("description","")

    if not ticket_id:
        logging.error("Ticket id not found in payload")
        return {"ok": False, "error": "ticket id not found"}

    # Compose a clear prompt asking for structured JSON output
    system = ("You are a customer support assistant. Return only valid JSON with keys: "
              "intent (one word), confidence (0-1), summary (2-3 lines in Hindi), "
              "sentiment (Angry/Neutral/Positive), reply_draft (friendly Hindi reply), "
              "kb_suggestions (list of short titles or URLs).")

    user_prompt = f"""Ticket subject:
{subject}

Ticket body:
{description}

Return only a valid JSON object. Example:
{{"intent":"BILLING","confidence":0.92,"summary":"...","sentiment":"Angry","reply_draft":"...","kb_suggestions":["KB1","KB2"]}}
"""

    try:
        ai_resp = call_openai(system, user_prompt)
        assistant_text = ai_resp["choices"][0]["message"]["content"]
        logging.info("OpenAI raw response: %s", assistant_text)

        # Try to parse JSON from model
        parsed = json.loads(assistant_text)
    except Exception as e:
        # fallback: if model returned plain text or parse failed
        logging.exception("OpenAI or JSON parse error: %s", e)
        assistant_text = assistant_text if 'assistant_text' in locals() else ""
        parsed = {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "summary": assistant_text[:500],
            "sentiment": "UNKNOWN",
            "reply_draft": assistant_text[:2000],
            "kb_suggestions": []
        }

    # Build the note body to post as private note for agent
    note = f"""**AI Assist (draft)**

**Intent:** {parsed.get('intent')}
**Confidence:** {parsed.get('confidence')}

**Sentiment:** {parsed.get('sentiment')}

**Summary:**
{parsed.get('summary')}

**Draft Reply (agent can edit & send):**
{parsed.get('reply_draft')}

**KB Suggestions:**
{json.dumps(parsed.get('kb_suggestions', []), ensure_ascii=False)}

_Notes: AI-generated draft — please review before sending._
"""
    try:
        res = post_freshdesk_note(ticket_id, note, private=True)
        logging.info("Posted note to Freshdesk ticket %s", ticket_id)
    except Exception as e:
        logging.exception("Failed to post note to Freshdesk: %s", e)
        return {"ok": False, "error": str(e)}

    return {"ok": True, "ticket": ticket_id, "ai": parsed}
