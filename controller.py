import subprocess, time

pairs = ["EUR_USD","GBP_USD","USD_JPY"]
procs = {}

while True:
    for p in pairs:
        if p not in procs:
            procs[p] = subprocess.Popen(["python","main.py","--pair",p])
    time.sleep(60)
