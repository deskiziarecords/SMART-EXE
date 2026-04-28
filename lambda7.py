class Lambda7:
    def __init__(self):
        self._last_prices = []

    def update(self, prices):
        self._last_prices = prices

    def valid(self, direction: str) -> bool:
        if len(self._last_prices) < 2:
            return True

        pct = (self._last_prices[-1] - self._last_prices[-2]) / self._last_prices[-2]

        # simple macro proxy
        if direction == "LONG" and pct < 0:
            return False
        if direction == "SHORT" and pct > 0:
            return False

        return True
