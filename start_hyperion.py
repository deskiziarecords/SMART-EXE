# start_hyperion.py
import subprocess
import time
import sys
import os
import webbrowser

def start_hyperion():
    print("Initializing Hyperion Sentinel System...")
    
    # 1. Start Backend
    print("Starting Backend Server (FastAPI)...")
    backend = subprocess.Popen([sys.executable, "main_server.py"], cwd=os.getcwd())
    
    # Wait for backend to warm up
    time.sleep(3)
    
    # 2. Start Frontend
    print("Starting Frontend Dashboard (Vite)...")
    ui_dir = os.path.join(os.getcwd(), "ui")
    frontend = subprocess.Popen(["npm", "run", "dev"], cwd=ui_dir, shell=True)
    
    # Wait for Vite to output the URL
    time.sleep(3)
    
    # 3. Open Browser
    print("Launching Control Surface...")
    webbrowser.open("http://localhost:5173")
    
    print("\nHyperion Sentinel is Running.")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("\nPress Ctrl+C to shutdown all systems.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
        print("Hyperion Sentinel Terminated.")

if __name__ == "__main__":
    start_hyperion()
