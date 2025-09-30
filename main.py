import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
KNOWLEDGE_BASE_CSV = os.getenv("KNOWLEDGE_BASE_CSV")  # optional CSV path

# --------------------------
# Logging Config
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI()

# --------------------------
# Helper Functions
# --------------------------
def extract_requester_name(payload: dict) -> str:
    ticket = payload.get("ticket") or payload
    if "requester" in ticket and "name" in ticket["requester"]:
        return ticket["requester"]["name"]
    if "contact" in ticket and "name" in ticket["contact"]:
        return ticket["contact"]["name"]
    if "requester_name" in ticket:
        return ticket["requester_name"]
    return "Customer"

def extract_from_pdf(file_path: str) -> str:
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error("Error extracting PDF: %s", e)
        return ""

def extract_from_csv(file_path: str, query_terms: list) -> str:
    try:
        df = pd.read_csv(file_path)
        content = ""
        for term in query_terms:
            filtered = df[df.apply(lambda row: row.astype(str).str.contains(term, case=False).any(), axis=1)]
            if not filtered.empty:
                content += filtered.to_csv(index=False) + "\n"
        return content
    except Exception as e:
        logging.error("Error extracting CSV: %s", e)
        return ""

def call_openai(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error("OpenAI API error: %s", e)
        return "Sorry, I couldn't generate a draft response at this time."

# --------------------------
# Create Draft in Freshdesk
# --------------------------
def create_freshdesk_draft(ticket_id: int, draft_message: str):
    """
    This function creates a private note in Freshdesk ticket
    so the agent can review and send manually.
    """
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes"
    auth = (FRESHDESK_API_KEY, "X")
    data = {
        "body": draft_message,
        "private": True  # ensures it's a draft note, not visible to requester
    }
    try:
        r = requests.post(url, auth=auth, json=data)
        r.raise_for_status()
        logging.info("Draft note created for ticket #%d", ticket_id)
    except Exception as e:
        logging.error("Error creating draft note in Freshdesk: %s", e)

# --------------------------
# Webhook Endpoint
# --------------------------
@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    payload = await request.json()
    logging.info("Webhook received: %s", json.dumps(payload))

    ticket_id = payload.get("ticket_id") or payload.get("ticket", {}).get("id")
    requester_name = extract_requester_name(payload)
    ticket_description = payload.get("ticket", {}).get("description_text", "")

    logging.info("Processing ticket #%s from %s", ticket_id, requester_name)

    # Extract CSV KB content
    kb_content = ""
    query_terms = ticket_description.split()[:5]  # simple term extraction
    if KNOWLEDGE_BASE_CSV and os.path.exists(KNOWLEDGE_BASE_CSV):
        kb_content = extract_from_csv(KNOWLEDGE_BASE_CSV, query_terms)
        logging.info("KB content length: %d", len(kb_content))

    # If KB is empty, be polite
    if not kb_content.strip():
        kb_content = "Currently, we do not have a direct knowledge base article for this query. Please respond professionally."

    # Prepare AI prompt
    prompt = f"""
You are a professional customer support agent.
Requester Name: {requester_name}
Ticket Description: {ticket_description}
Knowledge Base: {kb_content}

Generate a professional email draft to the customer including:
- Greeting
- Polite response using available KB info
- Signature
Keep it polite, professional, and helpful.
"""

    ai_response = call_openai(prompt)

    # Post as private draft note in Freshdesk
    if ticket_id:
        create_freshdesk_draft(ticket_id, ai_response)

    return JSONResponse({"status": "draft_created", "ticket_id": ticket_id})

# --------------------------
# Health Check / Root
# --------------------------
@app.get("/")
async def root():
    return {"status": "live", "message": "Email automation agent is running!"}
