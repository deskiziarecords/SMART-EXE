from config import ENTROPY_THRESHOLD
from logger import log_block

class RiskEngine:
    def evaluate(self, state):
        if state["entropy"] > ENTROPY_THRESHOLD:
            return self.block(state,"entropy")

        if abs(state["memory_bias"]) < 0.3:
            return self.block(state,"memory")

        if not state["lambda7_ok"]:
            return self.block(state,"lambda7")

        return {
            "action":"ALLOW",
            "size":0.01 + abs(state["memory_bias"])*0.04
        }

    def block(self,state,reason):
        log_block(state,reason)
        return {"action":"BLOCK"}
