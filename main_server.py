# main_server.py
import asyncio
import json
import random
import time
from collections import deque, Counter
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Hyperion Imports
from encoder import SymbolicEncoder, CandleType
from market_engine import MarketStateEngine
from tick_aggregator import MinuteAggregator, Tick
from lambda7 import MacroState

app = FastAPI()

# Allow CORS for UI development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HyperionServer:
    def __init__(self):
        self.encoder = SymbolicEncoder(lookback=20)
        self.market_engine = MarketStateEngine(window=60)
        self.connections: List[WebSocket] = []
        self.pattern_history = deque(maxlen=100)
        self.logs = deque(maxlen=20)
        self.current_price = 1.08500
        self.is_running = False
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "history": list(self.pattern_history),
            "logs": list(self.logs),
            "price": self.current_price
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for conn in self.connections:
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self.disconnect(conn)

    def on_new_candle(self, candle: dict):
        """Callback from aggregator"""
        symbol_type = self.encoder.encode_candle(
            candle['open'], candle['high'], 
            candle['low'], candle['close']
        )
        symbol = symbol_type.name
        self.pattern_history.append(symbol)
        
        # Update market engine
        state = self.market_engine.update(symbol)
        
        event = {
            "type": "update",
            "timestamp": candle['timestamp'],
            "symbol": symbol,
            "price": candle['close'],
            "state": state,
            "sequence": "".join(list(self.pattern_history)[-20:])
        }
        
        # Add log if action is interesting
        if state['action'] != 'HOLD':
            log_entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "msg": f"SIGNAL: {state['action']} (conf: {state['confidence']:.2f})",
                "type": "signal"
            }
            self.logs.append(log_entry)
            asyncio.create_task(self.broadcast({"type": "log", "data": log_entry}))

        asyncio.create_task(self.broadcast(event))

    async def run_mock_stream(self):
        """Simulates market data for UI testing"""
        self.is_running = True
        aggregator = MinuteAggregator(self.on_new_candle)
        print("Starting mock market stream...")
        
        while self.is_running:
            # Generate mock tick every 2 seconds (faster than real-time for demo)
            self.current_price += (random.random() - 0.495) * 0.0001
            tick = Tick(
                timestamp=time.time(),
                bid=self.current_price - 0.00005,
                ask=self.current_price + 0.00005,
                volume=random.randint(1, 10)
            )
            aggregator.ingest_tick(tick)
            
            # Broadcast price update more frequently than candles
            await self.broadcast({
                "type": "price",
                "price": round(self.current_price, 5),
                "timestamp": time.time()
            })
            
            await asyncio.sleep(1)

server_instance = HyperionServer()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(server_instance.run_mock_stream())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await server_instance.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client commands if needed
            msg = json.loads(data)
            if msg.get("command") == "toggle_mode":
                print(f"Mode toggled: {msg.get('mode')}")
    except WebSocketDisconnect:
        server_instance.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        server_instance.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
