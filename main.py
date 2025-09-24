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

# Configure Logging with File Handler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

# Load Environment Variables
load_dotenv()
FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "true").lower() == "true"
AUTO_REPLY_CONFIDENCE = float(os.getenv("AUTO_REPLY_CONFIDENCE", "0.80"))
SAFE_INTENTS = [i.strip().upper() for i in os.getenv("AUTO_REPLY_INTENTS", "COURSE_INQUIRY,GENERAL,INQUIRY").split(",")]
TEST_EMAIL = "komalsiddharth814@gmail.com"

if not (FRESHDESK_DOMAIN and FRESHDESK_API_KEY and OPENAI_API_KEY):
    logging.warning("❌ Missing required env vars: FRESHDESK_DOMAIN, FRESHDESK_API_KEY, OPENAI_API_KEY.")

# Initialize COURSES_DF
COURSES_FILE = "courses.csv"
if os.path.exists(COURSES_FILE):
    try:
        COURSES_DF = pd.read_csv(COURSES_FILE, encoding='utf-8', on_bad_lines='warn')
        logging.info("✅ Loaded COURSES_DF with %d rows (UTF-8)", len(COURSES_DF))
        COURSES_DF = COURSES_DF.fillna('')
        logging.info("📊 COURSES_DF columns: %s", COURSES_DF.columns.tolist())
        logging.info("📊 Sample courses: %s", COURSES_DF['Course Name'].head().tolist())
    except UnicodeDecodeError:
        try:
            COURSES_DF = pd.read_csv(COURSES_FILE, encoding='latin1', on_bad_lines='warn')
            logging.info("✅ Loaded COURSES_DF with %d rows (latin1 fallback)", len(COURSES_DF))
            COURSES_DF = COURSES_DF.fillna('')
            logging.info("📊 COURSES_DF columns: %s", COURSES_DF.columns.tolist())
            logging.info("📊 Sample courses: %s", COURSES_DF['Course Name'].head().tolist())
        except Exception as e:
            logging.error("❌ Failed to load COURSES_DF with fallback: %s", e)
            COURSES_DF = pd.DataFrame(columns=["Course Name"])
    except Exception as e:
        logging.error("❌ Failed to load COURSES_DF: %s", e)
        COURSES_DF = pd.DataFrame(columns=["Course Name"])
else:
    logging.info("ℹ️ courses.csv not found, using empty DataFrame")
    COURSES_DF = pd.DataFrame(columns=["Course Name"])

# App
app = FastAPI()

# Helper Functions
def call_openai(system_prompt: str, user_prompt: str, max_tokens=1000, temperature=0.1) -> dict:
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
    logging.info("🔍 API ticket response keys: %s", list(ticket_data.keys()))
    if 'requester' in ticket_data:
        logging.info("🔍 Requester details: %s", json.dumps(ticket_data['requester'], ensure_ascii=False))
    return ticket_data

def get_customer_name(ticket_data: dict) -> str:
    if 'requester' in ticket_data and isinstance(ticket_data['requester'], dict):
        contact = ticket_data['requester'].get('contact', {})
        name = contact.get('name', 'Customer')
        logging.info("👤 Extracted customer name: %s", name)
        return name
    logging.warning("⚠️ No customer name found, using 'Customer'")
    return "Customer"

def get_master_ticket_id(ticket_id: int) -> int:
    ticket = get_freshdesk_ticket(ticket_id)
    if not ticket:
        return ticket_id
    parent_id = ticket.get("merged_ticket_id") or ticket.get("custom_fields", {}).get("cf_parent_ticket_id")
    if parent_id:
        logging.info("🔀 Ticket %s merged into %s", ticket_id, parent_id)
        return get_master_ticket_id(parent_id)
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

def update_freshdesk_ticket_priority(ticket_id: int, priority: int = 3) -> dict:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}"
    payload = {"priority": priority}
    resp = requests.put(url, auth=(FRESHDESK_API_KEY, "X"), json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()

def extract_requester_email(payload: dict) -> str:
    logging.info("🔍 Full payload for email extraction: %s", json.dumps(payload, ensure_ascii=False, indent=2))
    ticket = payload.get("ticket") or payload
    logging.info("🔍 Ticket dictionary: %s", json.dumps(ticket, ensure_ascii=False, indent=2))

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

    try:
        email = payload["ticket"]["requester"]["email"].lower()
        logging.info("✅ Found email in nested ticket.requester.email")
        return email
    except Exception as e:
        logging.info("ℹ️ No direct/nested email found in payload (%s)", e)

    logging.info("ℹ️ No email found in payload, relying on API fetch")
    return ""

def get_course_details(course_name: str) -> dict:
    if not course_name or not isinstance(course_name, str):
        logging.info("ℹ️ Invalid or empty course_name: %s", course_name)
        return {"course_name": course_name, "fees": "", "link": "", "certificate": "", "notes": "", "details": "No matching course found"}

    try:
        matched = COURSES_DF[COURSES_DF['Course Name'].str.lower() == course_name.lower().strip()]
        if matched.empty:
            course_names = COURSES_DF['Course Name'].str.lower().tolist()
            fuzzy_matches = get_close_matches(course_name.lower().strip(), course_names, n=1, cutoff=0.6)
            if fuzzy_matches:
                matched = COURSES_DF[COURSES_DF['Course Name'].str.lower() == fuzzy_matches[0]]
                logging.info("🔍 Fuzzy matched '%s' to '%s'", course_name, fuzzy_matches[0])
        if not matched.empty:
            course_dict = matched.iloc[0].to_dict()
            course_dict.update({
                'fees': course_dict.get('Course Fees', ''),
                'link': course_dict.get('Course Link', ''),
                'certificate': course_dict.get('Course_Certificate', ''),
                'notes': course_dict.get('Notes', '')
            })
            logging.info("✅ Fetched full course details for '%s': %s", course_name, json.dumps(course_dict))
            return course_dict
        logging.warning("⚠️ No matching course found for '%s'. Available: %s", course_name, course_names)
        return {"course_name": course_name, "fees": "", "link": "", "certificate": "", "notes": "", "details": "No matching course found in our database. Please check the course name or contact support for more options."}
    except Exception as e:
        logging.exception("❌ Error in get_course_details for %s: %s", course_name, e)
        return {"course_name": course_name, "fees": "", "link": "", "certificate": "", "notes": "", "details": f"Error retrieving details: {str(e)}"}

def extract_possible_course(subject: str, description: str) -> str:
    patterns = [
        r"about\s+(.+?)(?:\s+course)?(?:\s|$)",
        r"course\s+(.+?)(?:\s+inquiry)?(?:\s|$)",
        r"inquiry\s+about\s+(.+?)(?:\s|$)",
        r"(?:health|wealth|relationship)\s+mastery",
        r"ho['’]oponopono\s+healer\s+certification",
        r"nlp\s+masterclass",
        r"loa\s+handwriting\s+frequency",
        r"platinum\s+membership",
        r"manifest\s+with\s+chakra",
        r"daily\s+magic\s+practise\s*\(?\s*dmp\s*\)?",
        r"advanced\s+law\s+of\s+attraction",
        r"(.+?)\s+course(?:\s|$)",
    ]
    text = f"{subject} {description}".lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            extracted = match.group(1).strip().title() if len(match.groups()) > 0 else match.group(0).title()
            logging.info("🔍 Course extraction matched pattern '%s' -> '%s'", pattern, extracted)
            return extracted
    # Fallback: Check for partial course name matches
    course_names = COURSES_DF['Course Name'].str.lower().tolist()
    words = text.split()
    for word in words:
        fuzzy_matches = get_close_matches(word, course_names, n=1, cutoff=0.6)
        if fuzzy_matches:
            logging.info("🔍 Fallback fuzzy matched '%s' to '%s'", word, fuzzy_matches[0])
            return fuzzy_matches[0].title()
    logging.info("🔍 No course pattern matched in text: %s", text[:200])
    return ""

def sanitize_dict(d: dict) -> dict:
    for k, v in list(d.items()):
        if isinstance(v, dict):
            d[k] = sanitize_dict(v)
        elif isinstance(v, float) and math.isnan(v):
            d[k] = None
        elif isinstance(v, list):
            d[k] = [sanitize_dict(item) if isinstance(item, dict) else item for item in v]
    return d

# Routes
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
        logging.info("📥 Incoming Freshdesk payload: ticket_id=%s, subject='%s'", 
                     payload.get("ticket", {}).get("id", "N/A"), payload.get("ticket", {}).get("subject", "N/A"))
        logging.info("📥 Full incoming payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as e:
        logging.exception("❌ Failed to parse JSON payload: %s", e)
        return {"ok": False, "error": "invalid JSON"}

    ticket = payload.get("ticket") or payload
    ticket_id = ticket.get("id") or payload.get("id")
    subject = ticket.get("subject", "")
    description = ticket.get("description", "")
    logging.info("🔹 Processing ticket %s: Subject='%s', Description preview='%s'", ticket_id, subject, description[:100])

    requester_email = extract_requester_email(payload)
    logging.info("🔹 Extracted ticket_id: %s, requester_email: %s", ticket_id, requester_email)

    if not ticket_id:
        logging.error("❌ Ticket id missing in payload")
        return {"ok": False, "error": "ticket id not found"}

    if not requester_email:
        logging.info("ℹ️ Requester email missing in payload, attempting API fetch")
        ticket_data = get_freshdesk_ticket(ticket_id)
        if ticket_data:
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

    try:
        master_id = get_master_ticket_id(ticket_id)
        logging.info("🔀 Master ticket id: %s (original: %s)", master_id, ticket_id)
    except Exception as e:
        logging.exception("❌ Failed to get master ticket id: %s", e)
        master_id = ticket_id

    ticket_data = get_freshdesk_ticket(master_id)
    customer_name = get_customer_name(ticket_data) if ticket_data else "Customer"

    possible_course = extract_possible_course(subject, description)
    logging.info("🔍 Extracted possible_course: '%s'", possible_course)
    if not possible_course:
        logging.info("ℹ️ No valid course name extracted from ticket %s", ticket_id)
        course_details = {"course_name": "", "fees": "", "link": "", "certificate": "", "notes": "", "details": "No course specified"}
    else:
        course_details = get_course_details(possible_course)
        logging.info("📚 Course details resolved: %s", json.dumps(course_details))

    system_prompt = """You are a customer support assistant for IMK team. Always respond in English only.

=== CLASSIFICATION RULES ===
- Classify as COURSE_INQUIRY if the query mentions a specific course name (e.g., 'Wealth Mastery', 'Health Mastery').
- If course details are provided, you MUST include ALL fields (Course Name, Course Fees, Course Link, Course_Certificate, Notes) in the reply_draft as bullet points in {{message_body}}. Do NOT say 'no info' if details exist.
- For course inquiries, set intent to COURSE_INQUIRY and confidence to 0.95.
- For payment-related queries (e.g., billing, payment issues), set intent to BILLING, confidence to 0.90, and include a note in {{message_body}} about escalating to the billing team.

=== OUTPUT FORMAT ===
Return valid JSON only with these keys:
- intent (one word: COURSE_INQUIRY, GENERAL, INQUIRY, BILLING)
- confidence (0-1 float, high for clear matches)
- summary (2-3 lines summarizing the query)
- sentiment (Angry/Neutral/Positive)
- reply_draft (MUST follow EXACT HTML format below with <br> for line breaks; insert course details as bullet points in {{message_body}})
- kb_suggestions (list of 3 short titles or URLs)

=== EXACT REPLY_DRAFT TEMPLATE (use placeholders) ===
<div>
  Hi {{customer_name}},<br><br>
  Thank you for reaching out.<br><br>
  {{message_body}}<br><br>
  Best regards,<br>
  IMK Support Team
</div>
"""

Course details provided: {course_details}. Integrate ALL fields as bullet points in {{message_body}} if query matches course_name (e.g., '<li>Course Name: Wealth Mastery</li><li>Fees: Rs. 15000</li><li>Link: https://miteshkhatri.com/JoinALOA</li><li>Certificate: No</li><li>Notes: None</li>'). For BILLING, include: 'Your query has been escalated to our billing team for prompt resolution.'""".format(course_details=json.dumps(course_details) if possible_course else "No specific course details available.")
    user_prompt = f"Ticket subject:\n{subject}\n\nTicket body:\n{description}\n\nCustomer Name: {customer_name}\n\nReturn valid JSON only."

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
            "reply_draft": f"Hi {customer_name},<br><br>Thank you for reaching out to us,<br><br>This is Rahul from Team IMK, we are here to help you.<br><br>AI parsing failed. Please review manually.<br><br>If you have any course-related technical questions, please email us at contact@miteshkhatri.com.<br>Our team is happy to help!<br><br>Thanks & Regards,<br>Rahul<br>Team IMK<br><img src='https://indattachment.freshdesk.com/inline/attachment?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA2MDAxNTMxMTAxOCwiZG9tYWluIjoibWl0ZXNoa2hhdHJpdHJhaW5pbmdsbHAuZnJlc2hkZXNrLmNvbSIsImFjY291bnRfaWQiOjMyMzYxMDh9.gswpN0f7FL4QfimJMQnCAKRj2APFqkOfYHafT0zB8J8' alt='IMK Team' style='width:200px;height:auto;'>",
            "kb_suggestions": []
        }

    intent = parsed.get("intent", "UNKNOWN").upper()
    confidence = float(parsed.get("confidence", 0.0))
    if math.isnan(confidence):
        confidence = 0.0
    logging.info("🧠 AI Classification: Intent=%s, Confidence=%.2f", intent, confidence)
    is_payment_issue = intent in ["BILLING", "PAYMENT"]

    note = f"""**🤖 AI Assist (draft)**

**Ticket ID:** {ticket_id} | **Master ID:** {master_id}
**Customer:** {customer_name} | **Email:** {requester_email}
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
    logging.info("📝 Generated draft note content: %s", note)
    try:
        post_freshdesk_note(master_id, note, private=True)
        logging.info("✅ Posted private draft to ticket %s", master_id)
    except Exception as e:
        logging.exception("❌ Failed posting note: %s", e)

    if is_payment_issue:
        try:
            update_freshdesk_ticket_priority(master_id, priority=3)
            logging.info("✅ Set ticket %s to high priority (billing issue)", master_id)
        except Exception as e:
            logging.exception("❌ Failed to set ticket priority: %s", e)
    else:
        auto_reply_ok = ENABLE_AUTO_REPLY and intent in SAFE_INTENTS and confidence >= AUTO_REPLY_CONFIDENCE
        logging.info("📤 Auto-reply check: Enabled=%s, Safe Intent=%s, Confidence OK=%s, Payment=%s -> OK=%s", 
                     ENABLE_AUTO_REPLY, intent in SAFE_INTENTS, confidence >= AUTO_REPLY_CONFIDENCE, is_payment_issue, auto_reply_ok)
        if auto_reply_ok:
            try:
                reply_body = parsed.get("reply_draft", f"Hi {customer_name},<br><br>Thank you for reaching out to us,<br><br>This is Rahul from Team IMK, we are here to help you.<br><br>Thank you for your inquiry. Our team will assist shortly.<br><br>If you have any course-related technical questions, please email us at contact@miteshkhatri.com.<br>Our team is happy to help!<br><br>Thanks & Regards,<br>Rahul<br>Team IMK<br><img src='https://indattachment.freshdesk.com/inline/attachment?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA2MDAxNTMxMTAxOCwiZG9tYWluIjoibWl0ZXNoa2hhdHJpdHJhaW5pbmdsbHAuZnJlc2hkZXNrLmNvbSIsImFjY291bnRfaWQiOjMyMzYxMDh9.gswpN0f7FL4QfimJMQnCAKRj2APFqkOfYHafT0zB8J8' alt='IMK Team' style='width:200px;height:auto;'>")
                logging.info("📤 Outgoing auto-reply body: %s", reply_body)
                post_freshdesk_reply(master_id, reply_body)
                logging.info("✅ Sent auto-reply to ticket %s", master_id)
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
        "auto_reply": not is_payment_issue and auto_reply_ok,
        "course_details": course_details,
        "possible_course": possible_course,
        "customer_name": customer_name
    }
    return sanitize_dict(response_data)

