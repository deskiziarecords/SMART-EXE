import json, os
from datetime import datetime

LOG_FILE = "logs/blocked_trades.jsonl"
TRADE_LOG = "logs/trades.jsonl"
SIGNAL_LOG = "logs/signals.jsonl"

os.makedirs("logs", exist_ok=True)

def log_block(state, reason):
    entry = {
        "time": datetime.utcnow().isoformat(),
        "type": "BLOCK",
        "reason": reason,
        "entropy": state.get("entropy"),
        "direction": state.get("direction"),
        "memory": state.get("memory_bias")
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def log_signal(state, decision):
    entry = {
        "time": datetime.utcnow().isoformat(),
        "type": "SIGNAL",
        "direction": decision.get("direction"),
        "size": decision.get("size"),
        "reason": decision.get("reason"),
        "entropy": state.get("entropy"),
        "memory_bias": state.get("memory_bias")
    }
    print(f"  [SIGNAL] {entry['direction']} {entry['size']} | {entry['reason']}")
    with open(SIGNAL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def log_trade(trade, price, pips):
    entry = {
        "time": datetime.utcnow().isoformat(),
        "type": "TRADE_CLOSE",
        "trade_id": trade.get("id"),
        "direction": trade.get("direction"),
        "entry": trade.get("entry"),
        "exit": price,
        "pips": pips,
        "sequence": trade.get("sequence")
    }
    print(f"  [TRADE] CLOSED {entry['direction']} {entry['pips']:+.1f}p")
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
