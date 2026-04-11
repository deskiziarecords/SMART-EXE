# tick_aggregator.py
import asyncio
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class Tick:
    timestamp: float
    bid: float
    ask: float
    volume: float

class MinuteAggregator:
    def __init__(self, on_candle_close):
        self.current_minute = None
        self.open = None
        self.high = -float('inf')
        self.low = float('inf')
        self.close = None
        self.volume = 0
        self.on_candle_close = on_candle_close  # Callback to pattern encoder
        
    def ingest_tick(self, tick: Tick):
        minute = int(tick.timestamp // 60)
        
        if minute != self.current_minute and self.current_minute is not None:
            # Close previous candle
            candle = {
                'timestamp': self.current_minute * 60,
                'open': self.open,
                'high': self.high,
                'low': self.low,
                'close': self.close,
                'volume': self.volume
            }
            self.on_candle_close(candle)
            
            # Reset
            self.open = None
        
        self.current_minute = minute
        
        if self.open is None:
            self.open = (tick.bid + tick.ask) / 2
            
        mid = (tick.bid + tick.ask) / 2
        self.high = max(self.high, mid)
        self.low = min(self.low, mid)
        self.close = mid
        self.volume += tick.volume
