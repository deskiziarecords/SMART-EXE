import requests, time
from config import *

def get_price():
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/pricing?instruments={PAIR}"
    r = requests.get(url,
        headers={"Authorization":f"Bearer {OANDA_API_KEY}"})
    return float(r.json()["prices"][0]["bids"][0]["price"])
