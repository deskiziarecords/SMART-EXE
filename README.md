# SMART-EXE
INDEPENDENT PORTABLE SOFTWARE
``` text
smartexe/
│
├── main.py
├── config.py
├── data_feed.py
├── encoder.py
├── memory.py
├── model.py
├── risk_engine.py
├── trader.py
├── lambda7.py
├── logger.py
├── dataset_builder.py
├── controller.py
│
└── logs/
    └── blocked_trades.jsonl
```
#  Hyperion Sentinel — Adelic-Koopman Micro Trading System

> A minimal, high-signal trading system focused on **pattern entropy, memory alignment, and causal validation** — designed for stealth, precision, and consistency.

---

##  Philosophy

Hyperion is not a “high-frequency noise machine.”

It trades only when:

* **Entropy is low** → market structure is predictable
* **Memory agrees** → historical patterns align
* **Causal signals confirm** → macro drivers support direction

> “No trade” is the default state. Precision is the edge.

---

##  Features

### Core Engine

* **Symbolic Encoding** — candles → letter sequences
*  **FAISS Memory** — pattern similarity + outcome bias
*  **Entropy Filter** — avoids chaotic regimes
*  **λ-System Risk Engine**

  * λ6: Micro sanity check
  * λ7: Macro causal trigger (e.g. DXY → EURUSD)

### Execution

*  OANDA live trading integration
*  Micro-lot stealth sizing (0.01–0.05)
*  Hard trade rejection logging

### Intelligence Layer

*  Dataset builder from real sessions
*  Neural model ready (GRU)
*  Self-evolution via blocked trade learning

---

##  Project Structure

```
hyperion/
│
├── main.py              # Core loop (live trading brain)
├── config.py            # API + parameters
├── data_feed.py         # OANDA pricing
├── encoder.py           # Candle → symbol
├── memory.py            # FAISS pattern memory
├── model.py             # Neural model (GRU)
├── risk_engine.py       # Trade gating logic
├── trader.py            # Order execution
├── lambda7.py           # Macro causal engine
├── logger.py            # Blocked trade logs
├── dataset_builder.py   # Training dataset
├── controller.py        # Multi-asset launcher
│
└── logs/
    └── blocked_trades.jsonl
```

---

## Installation

### 1. Clone / Setup

```bash
git clone <your-repo>
cd hyperion
```

### 2. Install Dependencies

```bash
pip install numpy pandas requests faiss-cpu torch
```

---

## Configuration

Edit `config.py`:

```python
OANDA_API_KEY = "YOUR_API_KEY"
ACCOUNT_ID = "YOUR_ACCOUNT_ID"
PAIR = "EUR_USD"
```

---

## Run the System

```bash
python main.py
```

---

## Safe Start (IMPORTANT)

Before using real money:

* Use **OANDA practice account**
* Run for **24–72 hours**
* Analyze logs:

```bash
python dataset_builder.py
```

---

## Risk Model

The system executes trades **ONLY IF ALL CONDITIONS PASS**:

| Gate         | Description                       |
| ------------ | --------------------------------- |
| Entropy      | Must be below threshold           |
| Memory Bias  | Pattern must have historical edge |
| λ7 Causality | Macro direction must agree        |

Otherwise:

→ trade is **BLOCKED and logged**

---

## Logging

All rejected trades are stored:

```
logs/blocked_trades.jsonl
```

Each entry includes:

* reason (entropy / memory / λ7)
* direction
* entropy value
* memory bias

---

## Training Pipeline

Future evolution:

1. Collect logs
2. Convert to dataset
3. Train neural model

Command:

```bash
python dataset_builder.py
```

---

## Roadmap

* [ ] Neural embedding upgrade (replace ASCII encoding)
* [ ] Auto-threshold adaptation (learn from blocked trades)
* [ ] Multi-asset orchestrator (session-aware launcher)
* [ ] GUI (Wireshark-style pattern visualization)
* [ ] MCP server (local + frontier model interaction)

---

## Disclaimer

This is an experimental system.

* Not financial advice
* Use at your own risk
* Always start with demo trading

---

## Core Idea

This system does not predict price.

It detects:

> **When the market becomes predictable enough to act.**

---

## Next Steps

After running:

 “train model from logs”
 “upgrade embedding to neural model”
 “build GUI analyzer”

---

## Final Note

Consistency > Frequency
Silence > Noise
Execution > Prediction

---

