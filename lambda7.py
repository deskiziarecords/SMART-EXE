import time

class Lambda7:
    def __init__(self):
        self.event = None

    def update(self, prices):
        if len(prices) < 5:
            return

        move = (prices[-1]-prices[0])/prices[0]

        if abs(move) > 0.002:
            self.event = {
                "dir": "UP" if move>0 else "DOWN",
                "time": time.time()
            }

    def valid(self, direction):
        if not self.event:
            return True

        if time.time() - self.event["time"] > 420:
            return True

        if self.event["dir"]=="UP" and direction=="SHORT":
            return True
        if self.event["dir"]=="DOWN" and direction=="LONG":
            return True

        return False
