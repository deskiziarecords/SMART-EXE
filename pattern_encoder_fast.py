# pattern_encoder_fast.py
import numba
import numpy as np

@numba.jit(nopython=True)
def encode_candle_numba(o, h, l, c, body_thresh=0.0001, wick_thresh=0.00015):
    """Sub-microsecond pattern encoding"""
    body = c - o
    body_size = abs(body)
    range_ = h - l
    
    if range_ == 0:
        return 0  # I
    
    if body_size <= 0.00005 or (body_size / range_ < 0.1):
        return 0  # I
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    if upper_wick > wick_thresh and upper_wick > body_size * 1.5:
        return 5  # W
    
    if lower_wick > wick_thresh and lower_wick > body_size * 1.5:
        return 6  # w
    
    is_strong = body_size > body_thresh
    
    if body > 0:
        return 1 if is_strong else 3  # B or U
    else:
        return 2 if is_strong else 4  # X or D
    
    return 0

SYMBOL_MAP = {0: 'I', 1: 'B', 2: 'X', 3: 'U', 4: 'D', 5: 'W', 6: 'w'}
COLOR_MAP = {
    'I': '#808080',  # Gray
    'B': '#00FF00',  # Bright green
    'U': '#90EE90',  # Light green
    'X': '#FF0000',  # Bright red
    'D': '#FFA500',  # Orange
    'W': '#FFD700',  # Gold (warning up)
    'w': '#FF69B4'   # Pink (warning down)
}
