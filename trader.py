import requests
from config import BASE_URL, ACCOUNT_ID, OANDA_API_KEY, PAIR

class OandaTrader:
    def execute(self, direction, size, price=None):
        units = int(size * 100000)
        if direction == "SELL" or direction == "SHORT":
            units = -abs(units)
        else:
            units = abs(units)

        url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders"

        data = {
            "order": {
                "instrument": PAIR,
                "units": str(units),
                "type": "MARKET"
            }
        }

        try:
            r = requests.post(url,
                headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
                json=data)
            print(f"Trade Execution: {direction} {size} units. Response: {r.text}")
            return r.json()
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None

    def get_equity(self):
        # In a real system, this would call OANDA API to get account balance
        # For now, return a default value or mock it
        return 10000.0
