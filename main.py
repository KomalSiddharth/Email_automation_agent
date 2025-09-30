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
    return "Unknown"

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
        return "Sorry, I couldn't generate a response."

def post_to_freshdesk(ticket_id: int, message: str):
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/reply"
    auth = (FRESHDESK_API_KEY, "X")
    data = {"body": message}
    try:
        r = requests.post(url, auth=auth, json=data)
        r.raise_for_status()
        logging.info("Reply sent to Freshdesk ticket %d", ticket_id)
    except Exception as e:
        logging.error("Error posting reply to Freshdesk: %s", e)

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

    # Example: Extract CSV KB content
    kb_content = ""
    query_terms = ticket_description.split()[:5]  # simple term extraction
    if KNOWLEDGE_BASE_CSV and os.path.exists(KNOWLEDGE_BASE_CSV):
        kb_content = extract_from_csv(KNOWLEDGE_BASE_CSV, query_terms)
        logging.info("Extracted KB content length: %d", len(kb_content))

    # Combine ticket + KB for OpenAI
    prompt = f"Requester: {requester_name}\nDescription: {ticket_description}\nKnowledge Base: {kb_content}\n\nRespond politely and professionally."
    ai_response = call_openai(prompt)

    # Post back to Freshdesk
    if ticket_id:
        post_to_freshdesk(ticket_id, ai_response)

    return JSONResponse({"status": "success", "ticket_id": ticket_id})

# --------------------------
# Health Check / Root
# --------------------------
@app.get("/")
async def root():
    return {"status": "live", "message": "Email automation agent is running!"}
