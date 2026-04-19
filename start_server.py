"""Hyphae HTTP server watchdog — auto-restarts on crash."""
import subprocess
import sys
import time
import os

HYPHAE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HYPHAE_DIR, "hyphae.log")
MAX_RESTARTS = 10
RESTART_WINDOW = 300  # seconds — reset crash counter after 5 min of stable running

def main():
    crashes = 0
    last_start = 0

    while crashes < MAX_RESTARTS:
        now = time.time()
        # Reset crash counter if it's been stable for a while
        if now - last_start > RESTART_WINDOW:
            crashes = 0

        last_start = now
        print(f"[watchdog] Starting Hyphae server (attempt {crashes + 1})...", flush=True)

        with open(LOG_FILE, "a") as log:
            log.write(f"\n--- Hyphae starting at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            proc = subprocess.Popen(
                [sys.executable, "-c", "from src.hyphae.server import run; run()"],
                cwd=HYPHAE_DIR,
                stdout=log,
                stderr=log,
            )
            proc.wait()
            log.write(f"\n--- Hyphae exited with code {proc.returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        crashes += 1
        if crashes < MAX_RESTARTS:
            wait = min(5 * crashes, 30)
            print(f"[watchdog] Hyphae crashed (exit {proc.returncode}). Restarting in {wait}s...", flush=True)
            time.sleep(wait)

    print(f"[watchdog] Too many crashes ({MAX_RESTARTS}). Giving up.", flush=True)

if __name__ == "__main__":
    main()
