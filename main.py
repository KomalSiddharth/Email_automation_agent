import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import pandas as pd
import re  # For improved course extraction
import math  # For math.isnan check

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "true").lower() == "true"  # Default to true for testing
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", "0.80"))  # Lowered to 0.8 for testing
SAFE_INTENTS = [i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL,INQUIRY").split(",")]  # Added INQUIRY
TEST_EMAIL = "komalsiddharth814@gmail.com"  # Only this email is processed

if not (FRESHDESK_DOMAIN and FRESHDESK_API_KEY and OPENAI_API_KEY):
    logging.warning("❌ Missing required env vars: FRESHDESK_DOMAIN, FRESHDESK_API_KEY, OPENAI_API_KEY.")

# --------------------------
# Initialize COURSES_DF
# --------------------------
COURSES_FILE = "courses.csv"
if os.path.exists(COURSES_FILE):
    try:
        COURSES_DF = pd.read_csv(COURSES_FILE, encoding='utf-8', on_bad_lines='warn')  # Warn and skip bad lines
        logging.info("✅ Loaded COURSES_DF with %d rows (UTF-8)", len(COURSES_DF))
        COURSES_DF = COURSES_DF.fillna('')  # Replace NaN with empty string to prevent JSON issues
        logging.info("📊 COURSES_DF columns: %s", COURSES_DF.columns.tolist())
        logging.info("📊 Sample courses: %s", COURSES_DF['Course Name'].head().tolist())
    except UnicodeDecodeError:
        try:
            COURSES_DF = pd.read_csv(COURSES_FILE, encoding='latin1', on_bad_lines='warn')  # Fallback encoding, skip bad lines
            logging.info("✅ Loaded COURSES_DF with %d rows (latin1 fallback)", len(COURSES_DF))
            COURSES_DF = COURSES_DF.fillna('')  # Replace NaN with empty string
            logging.info("📊 COURSES_DF columns: %s", COURSES_DF.columns.tolist())
            logging.info("📊 Sample courses: %s", COURSES_DF['Course Name'].head().tolist())
        except Exception as e:
            logging.error("❌ Failed to load COURSES_DF with fallback: %s", e)
            COURSES_DF = pd.DataFrame(columns=["Course Name"])  # Empty fallback
    except Exception as e:
        logging.error("❌ Failed to load COURSES_DF: %s", e)
        COURSES_DF = pd.DataFrame(columns=["Course Name"])  # Empty fallback
else:
    logging.info("ℹ️ courses.csv not found, using empty DataFrame")
    COURSES_DF = pd.DataFrame(columns=["Course Name"])  # Empty fallback

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
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}?include=requester"
    resp = requests.get(url, auth=(FRESHDESK_API_KEY, "X"), timeout=20)
    if resp.status_code != 200:
        logging.error("❌ Failed to fetch ticket %s: %s", ticket_id, resp.text)
        return None
    ticket_data = resp.json()
    logging.info("🔍 API ticket response: %s", json.dumps(ticket_data, ensure_ascii=False))
    return ticket_data

def get_master_ticket_id(ticket_id: int) -> int:
    ticket = get_freshdesk_ticket(ticket_id)
    if not ticket:
        return ticket_id
    parent_id = ticket.get("merged_ticket_id") or ticket.get("custom_fields", {}).get("cf_parent_ticket_id")
    if parent_id:
        logging.info("🔀 Ticket %s merged into %s", ticket_id, parent_id)
        return get_master_ticket_id(parent_id)  # Recursive to handle deep merges
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
    logging.info("🔍 Full payload for email extraction: %s", json.dumps(payload, ensure_ascii=False, indent=2))
    ticket = payload.get("ticket") or payload
    logging.info("🔍 Ticket dictionary: %s", json.dumps(ticket, ensure_ascii=False, indent=2))

    # Direct checks
    if "requester" in ticket and isinstance(ticket["requester"], dict) and "email" in ticket["requester"]:
        logging.info("✅ Found email in ticket.requester.email")
        return ticket["requester"]["email"].lower()
    if "contact" in ticket and isinstance(ticket["contact"], dict) and "email" in ticket["contact"]:
        logging.info("✅ Found email in ticket.contact.email")
        return ticket["contact"]["email"].lower()
    if "requester_email" in ticket:
        logging.info("✅ Found email in ticket.requester_email")
        return ticket["requester_email"].lower()
    if "email" in ticket:
        logging.info("✅ Found email in ticket.email")
        return ticket["email"].lower()
    if "from" in ticket:
        logging.info("✅ Found email in ticket.from")
        return ticket["from"].lower()
    if "from_email" in ticket:
        logging.info("✅ Found email in ticket.from_email")
        return ticket["from_email"].lower()
    if "email_address" in ticket:
        logging.info("✅ Found email in ticket.email_address")
        return ticket["email_address"].lower()
    if "custom_fields" in ticket and isinstance(ticket["custom_fields"], dict):
        for key, value in ticket["custom_fields"].items():
            if "email" in key.lower() and value:
                logging.info("✅ Found email in ticket.custom_fields.%s", key)
                return value.lower()

    # Fallback for nested structures
    try:
        email = payload["ticket"]["requester"]["email"].lower()
        logging.info("✅ Found email in nested ticket.requester.email")
        return email
    except Exception as e:
        logging.info("ℹ️ No direct/nested email found in payload (%s)", e)

    logging.info("ℹ️ No email found in payload, relying on API fetch")
    return ""

def get_course_details(course_name: str) -> dict:
    """
    Fetch course details from COURSES_DF based on course_name - Improved fuzzy matching
    """
    if not course_name or not isinstance(course_name, str):
        logging.info("ℹ️ Invalid or empty course_name: %s", course_name)
        return {"course_name": course_name, "details": "No matching course found"}

    try:
        # Normalize and search for exact or partial match (case-insensitive)
        norm_course = course_name.lower().strip()
        matched = COURSES_DF[COURSES_DF['Course Name'].str.lower().str.contains(norm_course, na=False, case=False)]
        if matched.empty:
            # Try exact match if partial fails
            matched = COURSES_DF[COURSES_DF['Course Name'].str.lower() == norm_course]
        if not matched.empty:
            course_dict = matched.iloc[0].to_dict()
            logging.info("✅ Found course details for '%s': %s", course_name, json.dumps(course_dict))
            return course_dict
        logging.warning("⚠️ No matching course found for '%s' in CSV. Available courses: %s", course_name, COURSES_DF['Course Name'].tolist())
        return {"course_name": course_name, "details": "No matching course found in our database. Please check the course name or contact support for more options."}
    except Exception as e:
        logging.exception("❌ Error in get_course_details for %s: %s", course_name, e)
        return {"course_name": course_name, "details": f"Error retrieving details: {str(e)}"}

def extract_possible_course(subject: str, description: str) -> str:
    """
    Improved extraction: Scan subject and description for course names (e.g., after 'about', 'course', or standalone).
    """
    patterns = [
        r"about\s+(.+?)(?:\s+course)?",  # e.g., "about Python Basics"
        r"course\s+(.+?)(?:\s+inquiry)?",  # e.g., "course Python Basics"
        r"inquiry\s+about\s+(.+)",  # e.g., "inquiry about AI Fundamentals"
        r"(health|wealth)\s+mastery",  # Specific for "Health/Wealth Mastery"
        r"(.+?)\s+course",  # General course name
    ]
    text = f"{subject} {description}".lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            extracted = match.group(1).strip().title() if len(match.groups()) > 0 else match.group(0).title()
            logging.info("🔍 Course extraction matched pattern '%s' -> '%s'", pattern, extracted)
            return extracted
    logging.info("🔍 No course pattern matched in text: %s", text[:200])
    return ""  # No match

def sanitize_dict(d: dict) -> dict:
    """Recursively replace NaN with None in dict to make JSON compliant."""
    for k, v in list(d.items()):
        if isinstance(v, dict):
            d[k] = sanitize_dict(v)
        elif isinstance(v, float) and math.isnan(v):
            d[k] = None
        elif isinstance(v, list):
            d[k] = [sanitize_dict(item) if isinstance(item, dict) else item for item in v]
    return d

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
        logging.info("📩 Incoming Freshdesk payload received: ticket_id=%s, subject='%s'", 
                     payload.get("ticket", {}).get("id", "N/A"), payload.get("ticket", {}).get("subject", "N/A"))
    except Exception as e:
        logging.exception("❌ Failed to parse JSON payload: %s", e)
        return {"ok": False, "error": "invalid JSON"}

    # Extract ticket details safely
    ticket = payload.get("ticket") or payload
    ticket_id = ticket.get("id") or payload.get("id")
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")
    logging.info("🔹 Processing ticket %s: Subject='%s', Description preview='%s'", ticket_id, subject, description[:100])

    # Extract requester email robustly
    requester_email = extract_requester_email(payload)
    logging.info("🔹 Extracted ticket_id: %s, requester_email: %s", ticket_id, requester_email)

    if not ticket_id:
        logging.error("❌ Ticket id missing in payload")
        return {"ok": False, "error": "ticket id not found"}

    if not requester_email:
        logging.info("ℹ️ Requester email missing in payload, attempting API fetch")
        ticket_data = get_freshdesk_ticket(ticket_id)
        if ticket_data:
            # Check requester and custom fields in API response
            if "requester" in ticket_data and isinstance(ticket_data["requester"], dict) and "email" in ticket_data["requester"]:
                requester_email = ticket_data["requester"]["email"].lower()
                logging.info("✅ Fetched email from API: %s", requester_email)
            elif "custom_fields" in ticket_data and isinstance(ticket_data["custom_fields"], dict):
                for key, value in ticket_data["custom_fields"].items():
                    if "email" in key.lower() and value:
                        requester_email = value.lower()
                        logging.info("✅ Fetched email from API custom_fields.%s: %s", key, requester_email)
            else:
                logging.info("ℹ️ Requester email not found in API response")
                return {"ok": True, "skipped": True, "reason": "missing requester_email"}
        else:
            logging.info("ℹ️ Requester email not found in API response, skipping auto-reply")
            return {"ok": True, "skipped": True, "reason": "missing requester_email"}

    if requester_email.lower() != TEST_EMAIL.lower():
        logging.info("⏭️ Ignored ticket %s from %s (not test email)", ticket_id, requester_email)
        return {"ok": True, "skipped": True}

    logging.info("✅ Processing test ticket %s from %s", ticket_id, requester_email)

    # Get master ticket ID (handles replies/threads)
    try:
        master_id = get_master_ticket_id(ticket_id)
        logging.info("🔀 Master ticket id: %s (original: %s)", master_id, ticket_id)
    except Exception as e:
        logging.exception("❌ Failed to get master ticket id: %s", e)
        master_id = ticket_id

    # Extract possible course (improved to scan description too)
    possible_course = extract_possible_course(subject, description)
    logging.info("🔍 Extracted possible_course: '%s'", possible_course)
    if not possible_course:
        logging.info("ℹ️ No valid course name extracted from ticket %s", ticket_id)
        course_details = {"course_name": "", "details": "No course specified"}
    else:
        course_details = get_course_details(possible_course)
        logging.info("📚 Course details resolved: %s", json.dumps(course_details))

    # AI classification - Improved prompt to enforce COURSE_INQUIRY for course queries and use details
    system_prompt = (
        "You are a customer support assistant. Always respond in English only. "
        "Classify as COURSE_INQUIRY if the query mentions a specific course name. "
        "If course details are provided, you MUST use them accurately in the reply_draft—do not say 'no info' if details exist. "
        "For course inquiries, set intent to COURSE_INQUIRY and confidence to 0.95+ if match is clear. "
        "Return JSON with: intent (one word: COURSE_INQUIRY, GENERAL, INQUIRY, BILLING), confidence (0-1 float, high for clear matches), summary (2-3 lines), "
        "sentiment (Angry/Neutral/Positive), reply_draft (polite email reply using template, include ALL course details if relevant), "
        "kb_suggestions (list of 3 short titles or URLs).\n"
        "Course details provided: {course_details}. Integrate fully into reply_draft if query matches course_name.\n"
        "Reply template:\n"
        "Dear [CustomerName],\n\n"
        "[Helpful AI reply with accurate course info, duration, fees, link, certificate if applicable]\n\n"
        "Best regards,\nSupport Team"
    ).format(course_details=json.dumps(course_details) if possible_course else "No specific course details available.")
    user_prompt = f"Ticket subject:\n{subject}\n\nTicket body:\n{description}\n\nReturn valid JSON only."

    try:
        ai_resp = call_openai(system_prompt, user_prompt)
        assistant_text = ai_resp["choices"][0]["message"]["content"].strip()
        logging.info("🤖 OpenAI raw response: %s", assistant_text)
        parsed = json.loads(assistant_text)
        logging.info("🤖 Parsed AI response: %s", json.dumps(parsed))
    except Exception as e:
        logging.exception("⚠️ OpenAI or JSON parse error: %s", e)
        parsed = {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "summary": description[:200],
            "sentiment": "UNKNOWN",
            "reply_draft": "AI parsing failed. Please review manually.",
            "kb_suggestions": []
        }

    intent = parsed.get("intent", "UNKNOWN").upper()
    confidence = float(parsed.get("confidence", 0.0))  # Ensure float
    if math.isnan(confidence):
        confidence = 0.0  # Safeguard against NaN
    logging.info("🧠 AI Classification: Intent=%s, Confidence=%.2f", intent, confidence)
    is_payment_issue = intent in ["BILLING", "PAYMENT"]

    # Build draft note with more details
    note = f"""**🤖 AI Assist (draft)**

**Ticket ID:** {ticket_id} | **Master ID:** {master_id}
**Intent:** {intent}
**Confidence:** {confidence}
**Sentiment:** {parsed.get('sentiment')}
**Summary:** {parsed.get('summary')}
**Extracted Course:** {possible_course}
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
    logging.info("📤 Auto-reply check: Enabled=%s, Safe Intent=%s, Confidence OK=%s, Payment=%s -> OK=%s", 
                 ENABLE_AUTO_REPLY, intent in SAFE_INTENTS, confidence >= AUTO_REPLY_CONFIDENCE, is_payment_issue, auto_reply_ok)
    if auto_reply_ok:
        try:
            reply_body = parsed.get("reply_draft", "Thank you for your inquiry. Our team will assist shortly. Best regards, Support Team")
            post_freshdesk_reply(master_id, reply_body)
            logging.info("✅ Auto-replied to ticket %s with body: %s", master_id, reply_body[:100] + "..." if len(reply_body) > 100 else reply_body)
        except Exception as e:
            logging.exception("❌ Failed posting auto-reply: %s", e)
    else:
        logging.info("ℹ️ Auto-reply skipped for ticket %s (intent: %s, confidence: %.2f, payment: %s)", 
                     master_id, intent, confidence, is_payment_issue)

    response_data = {
        "ok": True,
        "ticket": ticket_id,
        "master_ticket": master_id,
        "intent": intent,
        "confidence": confidence,
        "requester_email": requester_email,
        "auto_reply": auto_reply_ok,
        "course_details": course_details,
        "possible_course": possible_course
    }
    return sanitize_dict(response_data)
