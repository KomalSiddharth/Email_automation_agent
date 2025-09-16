# fetch_tickets.py
import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")

if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
    raise SystemExit("Please set FRESHDESK_DOMAIN and FRESHDESK_API_KEY in .env")

# Freshdesk uses basic auth with API key as username and 'X' as password
auth = (FRESHDESK_API_KEY, "X")
base_url = f"https://{FRESHDESK_DOMAIN}/api/v2"

def list_tickets(per_page=5):
    url = f"{base_url}/tickets?page=1&per_page={per_page}"
    r = requests.get(url, auth=auth, timeout=15)
    r.raise_for_status()
    return r.json()

def get_ticket(ticket_id):
    url = f"{base_url}/tickets/{ticket_id}"
    r = requests.get(url, auth=auth, timeout=15)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    print("Fetching latest tickets (sample)...")
    tickets = list_tickets()
    for t in tickets:
        print(f"ID: {t.get('id')} | Subject: {t.get('subject')} | Status: {t.get('status')}")
    # Optional: fetch a single ticket detail if tickets exist
    if tickets:
        tid = tickets[0].get("id")
        print("\nFetching details for ticket id:", tid)
        detail = get_ticket(tid)
        print("From:", detail.get("requester_id"), "Created_at:", detail.get("created_at"))
        print("Description (first 300 chars):\n", (detail.get("description_text") or detail.get("description") or "")[:300])
