# import subprocess
# import threading

# print('Make sure to set .env variables.')
# input('Also make sure you are in the Poetry shell before running this... (enter to start)')

# def run_app():
#     subprocess.run(['python3', 'app.py'])

# def run_streamlit():
#     subprocess.run(['streamlit', 'run', 'ui/main.py'])

# def run():
#     # Run each process in a separate thread
#     t1 = threading.Thread(target=run_app)
#     t2 = threading.Thread(target=run_streamlit)
#     t1.start()
#     t2.start()

# run()

import os
import sys
import time
import threading
import subprocess
from urllib.request import urlopen

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:9091")
DOCS_URL = f"{API_BASE}/docs"

def wait_for_backend(url, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status in (200, 301, 302, 404):  # any response means server is up
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def run_app():
    # Use the current Python interpreter (works on Windows/macOS/Linux)
    subprocess.run([sys.executable, "app.py"], check=True)

def run_streamlit():
    subprocess.run(["streamlit", "run", "ui/main.py"], check=True)

def main():
    print("Starting FastAPI backend...")
    t_api = threading.Thread(target=run_app, daemon=True)
    t_api.start()

    print("Waiting for backend to be ready...")
    if not wait_for_backend(DOCS_URL, timeout=45):
        print(f"Warning: backend not reachable at {DOCS_URL} yet, launching UI anyway.")

    print("Starting Streamlit UI...")
    run_streamlit()

if __name__ == "__main__":
    # Optional: remove the Poetry prompt
    # print('Make sure to set .env variables.')
    # input('Also make sure you are in the Poetry shell before running this... (enter to start)')
    main()
