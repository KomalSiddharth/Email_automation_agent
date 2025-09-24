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

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)

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
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", "0.80"))
SAFE_INTENTS = [
    i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL,INQUIRY").split(",")
]
TEST_EMAIL = "komalsiddharth814@gmail.com"

if not (FRESHDESK_DOMAIN and FRESHDESK_API_KEY and OPENAI_API_KEY):
    logging.warning("❌ Missing required env vars: FRESHDESK_DOMAIN, FRESHDESK_API_KEY, OPENAI_API_KEY.")

# --------------------------
# Load Courses Database
# --------------------------
COURSES_FILE = "courses.csv"
if os.path.exists(COURSES_FILE):
    try:
        COURSES_DF = pd.read_csv(COURSES_FILE, encoding="utf-8", on_bad_lines="warn").fillna("")
        logging.info("✅ Loaded courses.csv with %d rows", len(COURSES_DF))
    except UnicodeDecodeError:
        try:
            COURSES_DF = pd.read_csv(COURSES_FILE, encoding="latin1", on_bad_lines="warn").fillna("")
            logging.info("✅ Loaded courses.csv with latin1 fallback, %d rows", len(COURSES_DF))
        except Exception as e:
            logging.error("❌ Failed to load courses.csv: %s", e)
            COURSES_DF = pd.DataFrame(columns=["Course Name"])
else:
    logging.info("ℹ️ No courses.csv found, using empty DataFrame")
    COURSES_DF = pd.DataFrame(columns=["Course Name"])

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI()

# --------------------------
# Helper Functions
# --------------------------
def call_openai(system_prompt: str, user_prompt: str, max_tokens=1000, temperature=0.1) -> dict:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
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

def get_customer_name(ticket_data: dict) -> str:
    if "requester" in ticket_data and isinstance(ticket_data["requester"], dict):
        return ticket_data["requester"].get("name", "Customer")
    return "Customer"

def get_master_ticket_id(ticket_id: int) -> int:
    ticket = get_freshdesk_ticket(ticket_id)
    if not ticket:
        return ticket_id
    parent_id = ticket.get("merged_ticket_id") or ticket.get("custom_fields", {}).get("cf_parent_ticket_id")
    return get_master_ticket_id(parent_id) if parent_id else ticket_id

def extract_requester_email(payload: dict) -> str:
    ticket = payload.get("ticket") or payload
    for key in ["requester", "contact"]:
        if key in ticket and isinstance(ticket[key], dict) and "email" in ticket[key]:
            return ticket[key]["email"].lower()
    for key in ["requester_email", "email", "from", "from_email", "email_address"]:
        if key in ticket:
            return ticket[key].lower()
    if "custom_fields" in ticket:
        for k, v in ticket["custom_fields"].items():
            if "email" in k.lower() and v:
                return v.lower()
    return ""

def get_course_details(course_name: str) -> dict:
    if not course_name:
        return {"course_name": "", "fees": "", "link": "", "certificate": "", "notes": "", "details": "No course specified"}
    try:
        matched = COURSES_DF[COURSES_DF["Course Name"].str.lower() == course_name.lower()]
        if matched.empty:
            fuzzy = get_close_matches(course_name.lower(), COURSES_DF["Course Name"].str.lower().tolist(), n=1, cutoff=0.6)
            if fuzzy:
                matched = COURSES_DF[COURSES_DF["Course Name"].str.lower() == fuzzy[0]]
        if not matched.empty:
            row = matched.iloc[0].to_dict()
            return {
                "course_name": row.get("Course Name", ""),
                "fees": row.get("Course Fees", ""),
                "link": row.get("Course Link", ""),
                "certificate": row.get("Course_Certificate", ""),
                "notes": row.get("Notes", ""),
            }
        return {"course_name": course_name, "fees": "", "link": "", "certificate": "", "notes": "", "details": "Course not found"}
    except Exception as e:
        logging.error("❌ Error in get_course_details: %s", e)
        return {"course_name": course_name, "fees": "", "link": "", "certificate": "", "notes": "", "details": "Error retrieving course"}

def extract_possible_course(subject: str, description: str) -> str:
    text = f"{subject} {description}".lower()
    patterns = [
        r"(wealth|health|relationship)\s+mastery",
        r"ho['’]oponopono\s+healer\s+certification",
        r"nlp\s+masterclass",
        r"loa\s+handwriting\s+frequency",
        r"platinum\s+membership",
        r"manifest\s+with\s+chakra",
        r"daily\s+magic\s+practise\s*\(?\s*dmp\s*\)?",
        r"advanced\s+law\s+of\s+attraction",
        r"(.+?)\s+course",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return (m.group(1) if m.groups() else m.group(0)).title()
    # fuzzy match fallback
    for word in text.split():
        fuzzy = get_close_matches(word, COURSES_DF["Course Name"].str.lower().tolist(), n=1, cutoff=0.7)
        if fuzzy:
            return fuzzy[0].title()
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
    payload = await request.json()
    ticket = payload.get("ticket") or payload
    ticket_id = ticket.get("id")
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")
    requester_email = extract_requester_email(payload)

    if not ticket_id:
        return {"ok": False, "error": "ticket id missing"}
    if not requester_email or requester_email.lower() != TEST_EMAIL.lower():
        return {"ok": True, "skipped": True}

    master_id = get_master_ticket_id(ticket_id)
    ticket_data = get_freshdesk_ticket(master_id) or {}
    customer_name = get_customer_name(ticket_data)

    possible_course = extract_possible_course(subject, description)
    course_details = get_course_details(possible_course)

    system_prompt = """You are a customer support assistant for IMK team. Always respond in English only.
    === CLASSIFICATION RULES ===
    - If course found, intent=COURSE_INQUIRY, confidence=0.95
    - If billing/payment query, intent=BILLING, confidence=0.90
    - Else, intent=GENERAL
    === OUTPUT FORMAT ===
    Return strict JSON with:
    intent, confidence, summary, sentiment, reply_draft (HTML with <br>), kb_suggestions (3 items list)
    """

    user_prompt = f"Subject: {subject}\nDescription: {description}\nCustomer: {customer_name}\nCourse: {json.dumps(course_details)}"
    response = call_openai(system_prompt, user_prompt)
    logging.info("🤖 OpenAI raw response: %s", json.dumps(response, indent=2))

    return {"ok": True, "ticket_id": ticket_id, "ai_response": response}
