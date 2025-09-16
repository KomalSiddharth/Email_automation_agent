from fastapi import FastAPI, Request
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")

@app.get("/")
def home():
    return {"message": "AI Email Automation Backend Running"}

@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    data = await request.json()
    print("🎫 New ticket webhook received:")
    print(data)

    # Optional: call OpenAI API to draft reply
    # response = call_openai(data['description'])
    # print("AI Draft:", response)

    return {"status": "success", "message": "Webhook received"}
