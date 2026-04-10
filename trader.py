import requests
from config import *

class Trader:
    def order(self, direction, size):
        units = int(size * 100000)
        if direction=="SHORT":
            units = -units

        url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders"

        data = {
            "order":{
                "instrument":PAIR,
                "units":str(units),
                "type":"MARKET"
            }
        }

        r = requests.post(url,
            headers={"Authorization":f"Bearer {OANDA_API_KEY}"},
            json=data)

        print(r.text)
