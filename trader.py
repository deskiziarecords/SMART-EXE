import random
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config import PAIR, MOCK_MODE

class Trader:
    def order(self, direction: str, size: float) -> int:
        if MOCK_MODE or not MT5_AVAILABLE:
            ticket = random.randint(100000, 999999)
            print(f"  [MOCK TRADER] {direction} {size} lots @ MOCK_PRICE ticket={ticket}")
            return ticket

        tick   = mt5.symbol_info_tick(PAIR)
        price  = tick.ask if direction == 'LONG' else tick.bid
        action = mt5.ORDER_TYPE_BUY if direction == 'LONG' else mt5.ORDER_TYPE_SELL

        result = mt5.order_send({
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   PAIR,
            "volume":   round(size, 2),
            "type":     action,
            "price":    price,
            "deviation":10,
            "magic":    20250428,
            "comment":  "SMART-EXE",
            "type_time":mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        })

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"  [TRADER] Order failed: {result.retcode} {result.comment}")
            return -1

        print(f"  [TRADER] {direction} {size} lots @ {price:.5f} ticket={result.order}")
        return result.order

    def close(self, ticket: int):
        if MOCK_MODE or not MT5_AVAILABLE:
            print(f"  [MOCK TRADER] Closing ticket {ticket}")
            return

        position = None
        for pos in mt5.positions_get():
            if pos.ticket == ticket:
                position = pos
                break
        if not position:
            return

        tick   = mt5.symbol_info_tick(PAIR)
        price  = tick.bid if position.type == 0 else tick.ask
        action = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY

        mt5.order_send({
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    PAIR,
            "volume":    position.volume,
            "type":      action,
            "price":     price,
            "position":  ticket,
            "deviation": 10,
            "magic":     20250428,
            "comment":   "SMART-EXE close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        })
