import json
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for file paths
FAQ_PATH = "/opt/render/project/src/data/faq.pdf"
COURSES_PATH = "/opt/render/project/src/data/courses.csv"

class FreshdeskWebhook(BaseModel):
    ticket: dict
    actor: dict
    requester: dict
    eventType: str

def load_knowledge_base():
    kb_content = ""
    if os.path.exists(FAQ_PATH):
        logger.info("Loading FAQ from %s", FAQ_PATH)
        # Add logic to read and process faq.pdf (e.g., using PyPDF2)
        kb_content += "FAQ content loaded\n"
    else:
        logger.warning("⚠️ PDF file not found at: %s", FAQ_PATH)
    
    if os.path.exists(COURSES_PATH):
        logger.info("Loading courses from %s", COURSES_PATH)
        # Add logic to read and process courses.csv (e.g., using pandas)
        kb_content += "Courses content loaded\n"
    else:
        logger.warning("⚠️ CSV file not found at: %s", COURSES_PATH)
    
    return kb_content

def get_openai_response(query, kb_content):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # Adjust model as needed
            prompt=f"Query: {query}\nKnowledge Base: {kb_content}\nProvide a JSON response with intent, confidence, summary, sentiment, reply_draft, and kb_suggestions.",
            max_tokens=500
        )
        assistant_text = response.choices[0].text.strip()
        logger.info("🤖 OpenAI raw response: %s", assistant_text)
        
        # Validate JSON before parsing
        if not assistant_text:
            logger.error("⚠️ Empty response from OpenAI")
            return None
        
        # Ensure the response is valid JSON
        try:
            parsed = json.loads(assistant_text)
            return parsed
        except json.JSONDecodeError as e:
            logger.error("⚠️ JSON parse error: %s", str(e))
            logger.debug("Raw response: %s", assistant_text)
            return None
    except Exception as e:
        logger.error("⚠️ OpenAI API error: %s", str(e))
        return None

@app.post("/freshdesk-webhook")
async def freshdesk_webhook(webhook: FreshdeskWebhook, request: Request):
    logger.info("📩 Incoming Freshdesk payload: %s", webhook.dict())
    
    ticket_id = webhook.ticket.get("id")
    requester_email = webhook.requester.get("email")
    logger.info("🔹 Extracted ticket_id: %s, requester_email: %s", ticket_id, requester_email)
    
    # Check if email is a test email (modify as per your logic)
    test_emails = ["test@example.com"]  # Replace with actual test emails
    if requester_email not in test_emails:
        logger.info("⏭️ Ignored ticket %s from %s (not test email)", ticket_id, requester_email)
        return {"status": "ignored"}
    
    # Load knowledge base content
    kb_content = load_knowledge_base()
    
    # Prepare query for OpenAI
    query = f"{webhook.ticket.get('subject')} {webhook.ticket.get('description_text')}"
    
    # Get OpenAI response
    openai_response = get_openai_response(query, kb_content)
    
    if not openai_response:
        logger.warning("⚠️ No valid OpenAI response for query: %s", query)
        # Fallback response
        reply_draft = "Dear Customer,\n\nWe are unable to process your request at this time. Please contact support for further assistance.\n\nBest regards,\nSupport Team"
    else:
        reply_draft = openai_response.get("reply_draft", "No reply draft provided")
    
    # Post draft to Freshdesk (implement your Freshdesk API call here)
    logger.info("✅ Posted private draft to ticket %s", ticket_id)
    
    # Optionally send auto-reply (implement your Freshdesk API call here)
    logger.info("ℹ️ Auto-reply skipped (intent/setting)")
    
    return {"status": "success"}

@app.get("/")
async def root():
    return {"message": "Email Automation Agent is running"}
