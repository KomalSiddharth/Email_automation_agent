import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import pandas as pd
import re
import math
from difflib import get_close_matches

# ---------------------
# Logging Configuration
# ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

# ---------------------
# Load Environment Variables
# ---------------------
load_dotenv()
FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "true").lower() == "true"
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", "0.80"))
SAFE_INTENTS = [i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL,INQUIRY").split(",")]

# ---------------------
# Load Courses DataFrame
# ---------------------
COURSES_FILE = "courses.csv"
if os.path.exists(COURSES_FILE):
    try:
        COURSES_DF = pd.read_csv(COURSES_FILE, encoding='utf-8', on_bad_lines='warn').fillna('')
        logging.info("✅ Loaded COURSES_DF with %d rows", len(COURSES_DF))
    except Exception:
        COURSES_DF = pd.read_csv(COURSES_FILE, encoding='latin1', on_bad_lines='warn').fillna('')
        logging.info("✅ Loaded COURSES_DF with latin1 fallback, %d rows", len(COURSES_DF))
else:
    COURSES_DF = pd.DataFrame(columns=["Course Name"])
    logging.info("ℹ️ courses.csv not found, using empty DataFrame")

# ---------------------
# FastAPI App
# ---------------------
app = FastAPI()

# ---------------------
# Helper Functions
# ---------------------
def call_openai(system_prompt: str, user_prompt: str, max_tokens=1000, temperature=0.1) -> dict:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def get_freshdesk_ticket(ticket_id: int) -> dict | None:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}?include=requester"
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
    return get_master_ticket_id(parent_id) if parent_id else ticket_id

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

def update_freshdesk_ticket_priority(ticket_id: int, priority: int = 3) -> dict:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    resp = requests.put(url, auth=(FRESHDESK_API_KEY, "X"), json={"priority": priority}, timeout=20)
    resp.raise_for_status()
    return resp.json()

def extract_requester_email(ticket: dict) -> str:
    if "requester" in ticket and "email" in ticket["requester"]:
        return ticket["requester"]["email"].lower()
    return ""

def get_customer_name(ticket_data: dict) -> str:
    return ticket_data.get("requester", {}).get("name", "Customer") if ticket_data else "Customer"

def extract_possible_course(subject: str, description: str) -> str:
    text = f"{subject} {description}".lower()
    course_names = COURSES_DF['Course Name'].str.lower().tolist()
    for word in text.split():
        matches = get_close_matches(word, course_names, n=1, cutoff=0.6)
        if matches:
            return matches[0].title()
    return ""

def get_course_details(course_name: str) -> dict:
    if not course_name:
        return {"course_name": "", "fees": "", "link": "", "certificate": "", "notes": "", "details": "No course specified"}
    matched = COURSES_DF[COURSES_DF['Course Name'].str.lower() == course_name.lower()]
    if matched.empty:
        return {"course_name": course_name, "fees": "", "link": "", "certificate": "", "notes": "", "details": "No matching course found"}
    row = matched.iloc[0]
    return {
        "course_name": row.get("Course Name", ""),
        "fees": row.get("Course Fees", ""),
        "link": row.get("Course Link", ""),
        "certificate": row.get("Course_Certificate", ""),
        "notes": row.get("Notes", ""),
        "details": "Course found"
    }

def sanitize_dict(d: dict) -> dict:
    for k, v in list(d.items()):
        if isinstance(v, dict):
            d[k] = sanitize_dict(v)
        elif isinstance(v, float) and math.isnan(v):
            d[k] = None
        elif isinstance(v, list):
            d[k] = [sanitize_dict(i) if isinstance(i, dict) else i for i in v]
    return d

# ---------------------
# Routes
# ---------------------
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
        ticket = payload.get("ticket") or payload
        ticket_id = ticket.get("id")
        subject = ticket.get("subject", "")
        description = ticket.get("description", "")
        logging.info("📥 Incoming ticket %s | subject: %s", ticket_id, subject[:100])
    except Exception as e:
        logging.exception("❌ Failed to parse JSON: %s", e)
        return {"ok": False, "error": "invalid JSON"}

    requester_email = extract_requester_email(ticket)
    logging.info("👤 Requester email: %s", requester_email)
    
    # Process all tickets, no email filter
    logging.info("✅ Processing ticket %s from %s", ticket_id, requester_email)

    master_id = get_master_ticket_id(ticket_id)
    ticket_data = get_freshdesk_ticket(master_id)
    customer_name = get_customer_name(ticket_data)

    possible_course = extract_possible_course(subject, description)
    course_details = get_course_details(possible_course) if possible_course else get_course_details("")

    system_prompt = f"""You are a customer support assistant for IMK team. Always respond in English only.
Return valid JSON with keys: intent, confidence, summary, sentiment, reply_draft, kb_suggestions.
Course details provided: {json.dumps(course_details)}"""
    user_prompt = f"Ticket subject:\n{subject}\n\nTicket body:\n{description}\n\nCustomer Name: {customer_name}"

    try:
        ai_resp = call_openai(system_prompt, user_prompt)
        parsed = json.loads(ai_resp["choices"][0]["message"]["content"].strip())
    except Exception as e:
        logging.exception("⚠️ OpenAI or JSON parse failed: %s", e)
        parsed = {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "summary": description[:200],
            "sentiment": "UNKNOWN",
            "reply_draft": f"Hi {customer_name},<br><br>Thanks for reaching out. We will respond shortly.",
            "kb_suggestions": []
        }

    intent = parsed.get("intent", "UNKNOWN").upper()
    confidence = float(parsed.get("confidence", 0.0))
    is_payment_issue = intent in ["BILLING", "PAYMENT"]

    # Post private draft note
    note_content = f"""** AI Assist (draft)**
Ticket ID: {ticket_id} | Master ID: {master_id}
Customer: {customer_name} | Email: {requester_email}
Intent: {intent} | Confidence: {confidence}
Summary: {parsed.get('summary')}
Course: {possible_course}
Course Details: {json.dumps(course_details, ensure_ascii=False)}
Draft Reply: {parsed.get('reply_draft')}
KB Suggestions: {json.dumps(parsed.get('kb_suggestions', []), ensure_ascii=False)}
{"⚠️ Payment-related issue → private draft only." if is_payment_issue else "_Note: AI draft — please review before sending._"}
"""
    try:
        post_freshdesk_note(master_id, note_content, private=True)
        logging.info("✅ Posted private draft to ticket %s", master_id)
    except Exception as e:
        logging.exception("❌ Failed to post note: %s", e)

    # Auto-reply if enabled and safe
    auto_reply_ok = ENABLE_AUTO_REPLY and intent in SAFE_INTENTS and confidence >= AUTO_REPLY_CONFIDENCE and not is_payment_issue
    if auto_reply_ok:
        try:
            post_freshdesk_reply(master_id, parsed.get("reply_draft", f"Hi {customer_name},<br><br>Thank you for reaching out."))
            logging.info("✅ Sent auto-reply for ticket %s", master_id)
        except Exception as e:
            logging.exception("❌ Failed to send auto-reply: %s", e)

    response_data = {
        "ok": True,
        "ticket": ticket_id,
        "master_ticket": master_id,
        "intent": intent,
        "confidence": confidence,
        "requester_email": requester_email,
        "auto_reply": auto_reply_ok,
        "course_details": course_details,
        "possible_course": possible_course,
        "customer_name": customer_name
    }
    return sanitize_dict(response_data)
