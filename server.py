from fastapi import FastAPI, Request
import uvicorn
import os
import logging
import json
from main import freshdesk_webhook  # Import the webhook function from main.py

# --------------------------
# App & Logging
# --------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@app.get("/")
def root():
    return {"message": "AI Email Automation Backend Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------
# Webhook Endpoint
# --------------------------
@app.post("/freshdesk-webhook")
async def webhook_endpoint(request: Request):
    """
    This endpoint receives Freshdesk ticket webhooks and processes them
    using the logic defined in main.py.
    """
    try:
        response = await freshdesk_webhook(request)
        logging.info("✅ Webhook processed successfully")
        return response
    except Exception as e:
        logging.exception("❌ Error processing webhook: %s", e)
        return {"ok": False, "error": str(e)}

# --------------------------
# Run the app
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides the correct port
    uvicorn.run(app, host="0.0.0.0", port=port)
