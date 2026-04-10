import json, os
from datetime import datetime

LOG_FILE = "logs/blocked_trades.jsonl"

os.makedirs("logs", exist_ok=True)

def log_block(state, reason):
    entry = {
        "time": datetime.utcnow().isoformat(),
        "reason": reason,
        "entropy": state["entropy"],
        "direction": state["direction"],
        "memory": state["memory_bias"]
    }
    with open(LOG_FILE,"a") as f:
        f.write(json.dumps(entry)+"\n")
