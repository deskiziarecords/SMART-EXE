import pandas as pd

df = pd.read_json("logs/blocked_trades.jsonl", lines=True)
print(df["reason"].value_counts())
