from dataclasses import dataclass

@dataclass
class MacroState:
    dxy_change: float
    dxy_trend: float
    spx_change: float
    yields_change: float

class Lambda7:
    def __init__(self):
        self._last_prices = []

    def update(self, prices):
        self._last_prices = prices

    def valid(self, direction: str) -> bool:
        if len(self._last_prices) < 2:
            return True
        pct = (self._last_prices[-1] - self._last_prices[-2]) / self._last_prices[-2]
        if direction == "LONG" and pct < 0:
            return False
        if direction == "SHORT" and pct > 0:
            return False
        return True

class Lambda7Engine:
    """Hyperion-compatible Lambda7 engine."""
    def validate_direction(self, direction, macro: MacroState):
        valid = True
        strength = 0.5

        # Logic from main_server.py suggests DXY conflict check
        if direction == "BUY" and macro.dxy_change > 0.05:
            valid = False
        elif direction == "SELL" and macro.dxy_change < -0.05:
            valid = False

        return valid, strength
