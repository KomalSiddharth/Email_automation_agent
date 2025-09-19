from fastapi import FastAPI, Request
import uvicorn
import os

app = FastAPI()

@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    data = await request.json()
    print("🎫 New ticket webhook received:")
    print(data)
    return {"status": "success", "message": "Webhook received"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render will provide the correct PORT
    uvicorn.run(app, host="0.0.0.0", port=port)

