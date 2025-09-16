from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/freshdesk-webhook")
async def freshdesk_webhook(request: Request):
    data = await request.json()
    print("🎫 New ticket webhook received:")
    print(data)
    return {"status": "success", "message": "Webhook received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
