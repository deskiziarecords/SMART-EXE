# main_server.py - Fully Connected Hyperion Brain
import asyncio
import json
import random
import time
from collections import deque, Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Hyperion Core Imports
from encoder import SymbolicEncoder, CandleType
from memory import FAISSMemory
from risk_engine import RiskEngine, MarketState
from lambda7 import Lambda7Engine, MacroState
from tick_aggregator import MinuteAggregator, Tick
from market_engine import MarketStateEngine # Legacy engine for comparison
from trader import OandaTrader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HyperionBrain:
    """Orchestrates the full analysis stack for the server"""
    def __init__(self):
        self.encoder = SymbolicEncoder(lookback=20)
        self.memory = FAISSMemory(dim=16, ngram_size=5)
        self.risk = RiskEngine(adaptive_thresholds=True)
        self.lambda7 = Lambda7Engine()
        self.session_symbols = []
        self.symbol_history = []
        
    def process_candle(self, candle: dict, macro: MacroState) -> Tuple[dict, dict]:
        # 1. Encode
        symbol_type = self.encoder.encode_candle(
            candle['open'], candle['high'], 
            candle['low'], candle['close']
        )
        symbol_name = symbol_type.name
        self.session_symbols.append(symbol_type.value)
        self.symbol_history.append(symbol_name)
        
        if len(self.session_symbols) < 10:
            return {"symbol": symbol_name, "action": "WARMUP"}, {}

        # 2. Compute Market State M
        rho_c = self.encoder.compute_rho_c_fixed(self.session_symbols)
        Q_session = self.encoder.compute_Q_session_fixed(self.session_symbols)
        
        # Trend efficiency
        k_bar = self.compute_avg_run(self.session_symbols)
        eta_trend = self.encoder.compute_eta_trend(k_bar, 1, 15)
        
        # Information Theory metrics
        I_context = self.memory.compute_context_mi(self.session_symbols)
        
        # Reversal density & Stop Hunt (using components from both engines)
        delta_r = self.compute_range_ext(candle)
        P_stop_hunt = self.detect_stop_hunt(self.session_symbols, candle)
        
        M = MarketState(
            rho_c=rho_c, delta_r=delta_r, P_stop_hunt=P_stop_hunt,
            eta_adj=eta_trend, # simplified
            eta_trend=eta_trend, Q_session=Q_session, I_context=I_context
        )
        
        # 3. Memory Bias
        expected_eps, confidence = self.memory.get_bias_and_confidence(self.session_symbols)
        
        # 4. Risk Evaluation
        should_trade, direction, block_reason = self.risk.evaluate(M, expected_eps, confidence)
        
        # 5. Causal Check
        if should_trade:
            valid, causal_strength = self.lambda7.validate_direction(direction, macro)
            if not valid:
                should_trade = False
                block_reason = "λ7_REJECT: DXY Conflict"
        
        # 6. Memory Update
        self.memory.add(self.session_symbols, symbol_type.value, expected_eps)
        self.risk.update_thresholds(M)
        
        # Prepare state for UI
        state_for_ui = {
            "action": direction if should_trade else "HOLD",
            "confidence": confidence,
            "block_reason": block_reason,
            "state_vector": {
                "rho_c": rho_c,
                "delta_r": delta_r,
                "P_stop": P_stop_hunt,
                "eta_trend": eta_trend,
                "Q_session": Q_session,
                "entropy": I_context # UI uses entropy label for MI
            }
        }
        
        return {"symbol": symbol_name, "action": state_for_ui["action"]}, state_for_ui

    def compute_avg_run(self, symbols):
        if not symbols: return 1.0
        runs = []
        count = 1
        for i in range(1, len(symbols)):
            if (symbols[i] in [0, 2] and symbols[i-1] in [0, 2]) or (symbols[i] in [1, 3] and symbols[i-1] in [1, 3]):
                count += 1
            else:
                if count > 1: runs.append(count)
                count = 1
        return np.mean(runs) if runs else 1.0

    def compute_range_ext(self, candle):
        body = abs(candle['close'] - candle['open'])
        avg_body = np.mean(self.encoder.avg_body_history) if self.encoder.avg_body_history else 0.0001
        return body / avg_body if avg_body > 0 else 1.0

    def detect_stop_hunt(self, symbols, candle):
        if len(symbols) < 2: return 0.0
        last, curr = symbols[-2], symbols[-1]
        if last in [0, 1] and curr in [4, 5]: return 0.75
        return 0.1

class HyperionServer:
    def __init__(self):
        self.brain = HyperionBrain()
        self.trader = OandaTrader()
        self.connections: List[WebSocket] = []
        self.logs = deque(maxlen=20)
        self.current_price = 1.08500
        self.is_running = False
        self.mode = "manual"
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        await websocket.send_json({
            "type": "init",
            "history": self.brain.symbol_history[-100:],
            "logs": list(self.logs),
            "price": self.current_price
        })

    async def broadcast(self, message: dict):
        for conn in self.connections:
            try: await conn.send_json(message)
            except: self.connections.remove(conn)

    def on_new_candle(self, candle: dict):
        # Mock macro data
        macro = MacroState(
            dxy_change=(random.random() - 0.5) * 0.1,
            dxy_trend=random.random(),
            spx_change=(random.random() - 0.5) * 0.2,
            yields_change=(random.random() - 0.5) * 0.05
        )
        
        summary, state = self.brain.process_candle(candle, macro)
        
        event = {
            "type": "update",
            "timestamp": candle['timestamp'],
            "symbol": summary['symbol'],
            "price": candle['close'],
            "state": state,
            "sequence": "".join(self.brain.symbol_history[-20:])
        }
        
        if state and state['action'] != 'HOLD':
            log_entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": f"SIGNAL: {state['action']} confirmed at {candle['close']:.5f}", "type": "signal"}
            if self.mode == "auto":
                # Execute mock trade
                self.trader.execute(state['action'], 0.01, candle['close'])
                log_entry['msg'] += " [AUTO-EXECUTED]"
            
            self.logs.append(log_entry)
            asyncio.create_task(self.broadcast({"type": "log", "data": log_entry}))
        elif state and state['block_reason']:
             log_entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": f"BLOCKED: {state['block_reason']}", "type": "warn"}
             self.logs.append(log_entry)
             asyncio.create_task(self.broadcast({"type": "log", "data": log_entry}))

        asyncio.create_task(self.broadcast(event))

    async def run_market_loop(self):
        self.is_running = True
        aggregator = MinuteAggregator(self.on_new_candle)
        while self.is_running:
            self.current_price += (random.random() - 0.498) * 0.00015
            tick = Tick(time.time(), self.current_price-0.00004, self.current_price+0.00004, random.randint(1,15))
            aggregator.ingest_tick(tick)
            await self.broadcast({"type": "price", "price": round(self.current_price, 5)})
            await asyncio.sleep(0.5) # Fast demo mode (1 tick = ~0.5s, candle = 60s/tick_rate)

server_instance = HyperionServer()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(server_instance.run_market_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await server_instance.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("command") == "toggle_mode":
                server_instance.mode = msg.get("mode", "manual")
                print(f"System Mode: {server_instance.mode}")
    except WebSocketDisconnect:
        server_instance.connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
