# oanda_stream.py
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from tick_aggregator import Tick

class OandaStreamer:
    def __init__(self, account_id, api_key, aggregator):
        self.account_id = account_id
        self.api_key = api_key
        self.aggregator = aggregator
        self.base_url = "https://stream-fxpractice.oanda.com/v3/accounts"
        
    async def stream(self, instrument="EUR_USD"):
        url = f"{self.base_url}/{self.account_id}/pricing/stream"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"instruments": instrument}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                async for line in resp.content:
                    if line:
                        data = json.loads(line)
                        if data.get("type") == "PRICE":
                            tick = Tick(
                                timestamp=time.time(),
                                bid=float(data["bids"][0]["price"]),
                                ask=float(data["asks"][0]["price"]),
                                volume=float(data.get("tradeableUnits", 0))
                            )
                            self.aggregator.ingest_tick(tick)
