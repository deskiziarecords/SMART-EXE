SYMBOL_MAP = {0:'I',1:'B',2:'X',3:'U',4:'D',5:'W',6:'w'}
MAP = {'I':0,'B':1,'X':2,'U':3,'D':4,'W':5,'w':6}

def encode_candle(o,h,l,c):
    body = c - o
    rng = h - l + 1e-10

    if abs(body)/rng < 0.1:
        return 2  # X

    if body > 0:
        return 1 if abs(body) > 0.0001 else 3
    else:
        return 2 if abs(body) > 0.0001 else 4
