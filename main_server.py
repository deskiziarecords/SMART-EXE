# main_server.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatternStreamer:
    def __init__(self):
        self.connections = []
        self.pattern_history = deque(maxlen=1000)  # Last 1000 patterns
        self.aggregator = MinuteAggregator(self.on_new_candle)
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        # Send initial history
        await websocket.send_json({
            "type": "history",
            "patterns": list(self.pattern_history),
            "metrics": self.compute_metrics()
        })
        
    async def disconnect(self, websocket: WebSocket):
        self.connections.remove(websocket)
        
    def on_new_candle(self, candle):
        """Called every minute when candle closes"""
        pattern_idx = encode_candle_numba(
            candle['open'], candle['high'],
            candle['low'], candle['close']
        )
        pattern = SYMBOL_MAP[pattern_idx]
        
        event = {
            "type": "pattern",
            "timestamp": candle['timestamp'],
            "pattern": pattern,
            "ohlc": candle,
            "sequence": ''.join(self.pattern_history)[-20:] + pattern,
            "metrics": self.compute_metrics()
        }
        
        self.pattern_history.append(pattern)
        
        # Broadcast to all clients
        asyncio.create_task(self.broadcast(event))
        
    async def broadcast(self, message):
        disconnected = []
        for conn in self.connections:
            try:
                await conn.send_json(message)
            except:
                disconnected.append(conn)
        
        for conn in disconnected:
            self.connections.remove(conn)
    
    def compute_metrics(self):
        """Real-time entropy, mechanical score"""
        recent = list(self.pattern_history)[-60:]
        if not recent:
            return {}
        
        dist = Counter(recent)
        total = len(recent)
        
        from math import log2
        entropy = -sum((c/total)*log2(c/total) for c in dist.values())
        
        return {
            "entropy": round(entropy, 3),
            "mechanical_score": round(1 - dist.get('I', 0)/total, 3),
            "last_20": ''.join(recent)[-20:],
            "distribution": dict(dist)
        }

streamer = PatternStreamer()

@app.websocket("/ws/patterns")
async def websocket_endpoint(websocket: WebSocket):
    await streamer.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("action") == "query_rag":
                # Client requests RAG prediction
                sequence = msg.get("sequence", "")
                prediction = await query_rag(sequence)  # Your RAG system
                await websocket.send_json({
                    "type": "rag_prediction",
                    "prediction": prediction
                })
                
    except Exception as e:
        print(f"Client disconnected: {e}")
    finally:
        await streamer.disconnect(websocket)
