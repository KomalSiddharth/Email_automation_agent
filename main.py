import os
import json
import logging
import requests
from fastapi import FastAPI, Request
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
KNOWLEDGE_BASE_CSV = os.getenv("KNOWLEDGE_BASE_CSV", "")

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def extract_requester_email(payload: dict) -> str:
    ticket = payload.get("ticket") or payload
    if "requester" in ticket and "email" in ticket["requester"]:
        return ticket["requester"]["email"]
    if "contact" in ticket and "email" in ticket["contact"]:
        return ticket["contact"]["email"]
    if "requester_email" in ticket:
        return ticket["requester_email"]
    return ""

def extract_ticket_subject(payload: dict) -> str:
    ticket = payload.get("ticket") or payload
    return ticket.get("subject", "No Subject")

def extract_ticket_description(payload: dict) -> str:
    ticket = payload.get("ticket") or payload
    return ticket.get("description", "")

def extract_from_csv(csv_file: str, query_terms: list) -> str:
    if not csv_file or not os.path.exists(csv_file):
        logging.warning("⚠️ CSV file not found: %s", csv_file)
        return ""
    try:
        df = pd.read_csv(csv_file)
        kb_text = ""
        for term in query_terms:
            matches = df.apply(lambda row: term.lower() in row.to_string().lower(), axis=1)
            for i, match in enumerate(matches):
                if match:
                    kb_text += "\n" + df.iloc[i].to_string()
        return kb_text
    except Exception as e:
        logging.error(f"❌ Error reading CSV: {e}")
        return ""

def call_openai_api(prompt: str) -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        assistant_text = response.choices[0].message.get("content", "").strip()
        return assistant_text
    except Exception as e:
        logging.error(f"❌ OpenAI API call failed: {e}")
        return ""

def send_freshdesk_reply(ticket_id: int, reply_text: str) -> None:
    url = f"https://{FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/reply"
    auth = (FRESHDESK_API_KEY, "X")
    payload = {"body": reply_text}
    try:
        response = requests.post(url, auth=auth, json=payload)
        if response.status_code in [200, 201]:
            logging.info("✅ Reply sent successfully to ticket %s", ticket_id)
        else:
            logging.error("❌ Failed to send reply. Status code: %s, Response: %s",
                          response.status_code, response.text)
    except Exception as e:
        logging.error(f"❌ Exception sending reply: {e}")

# --------------------------
# Webhook Endpoint
# --------------------------
@app.post("/freshdesk_webhook")
async def freshdesk_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        logging.error(f"❌ Failed to parse JSON payload: {e}")
        raw_body = await request.body()
        logging.info(f"Raw body: {raw_body}")
        return {"status": "failed", "reason": "invalid JSON"}

    ticket_id = payload.get("ticket", {}).get("id") or payload.get("id")
    if not ticket_id:
        logging.warning("⚠️ Ticket ID not found in payload.")
        return {"status": "failed", "reason": "no ticket ID"}

    requester_name = extract_requester_name(payload)
    requester_email = extract_requester_email(payload)
    subject = extract_ticket_subject(payload)
    description = extract_ticket_description(payload)

    logging.info(f"📩 Received ticket #{ticket_id} from {requester_name} ({requester_email})")

    # Optionally extract CSV knowledge base content
    kb_content = ""
    if KNOWLEDGE_BASE_CSV:
        query_terms = description.split()[:5]  # Example: first 5 words as query terms
        kb_content = extract_from_csv(KNOWLEDGE_BASE_CSV, query_terms)
        if kb_content:
            logging.info("📚 Extracted KB content length: %d", len(kb_content))
        else:
            logging.warning("⚠️ No KB content extracted")

    # Prepare prompt for OpenAI
    prompt = f"""
    You are a professional support assistant.
    Ticket Subject: {subject}
    Ticket Description: {description}
    Requester Name: {requester_name}
    Requester Email: {requester_email}
    Knowledge Base: {kb_content}
    Write a professional and polite reply to the ticket.
    """

    assistant_text = call_openai_api(prompt)
    if not assistant_text:
        assistant_text = "Hello, we have received your ticket and will get back to you shortly."

    # Send reply
    send_freshdesk_reply(ticket_id, assistant_text)

    return {"status": "success", "ticket_id": ticket_id}
