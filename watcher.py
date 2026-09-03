"""Watches jira_tickets/ and spawns an independent `python main.py` process per new file."""

import os
import subprocess
import sys
import time

WATCH_DIR = "jira_tickets"
POLL_INTERVAL = 1.0  # seconds


def watch(directory: str = WATCH_DIR, interval: float = POLL_INTERVAL) -> None:
    os.makedirs(directory, exist_ok=True)
    seen = set(os.listdir(directory))
    print(f"👀 Watching '{directory}' for new tickets...")

    while True:
        time.sleep(interval)
        current = set(os.listdir(directory))
        for new_file in sorted(current - seen):
            path = os.path.join(directory, new_file)
            print(f"🆕 New ticket detected: {path} — spawning independent process")
            # Popen (not run) + no wait -> fire-and-forget, fully parallel processes.
            subprocess.Popen([sys.executable, "main.py", path])
        seen = current


if __name__ == "__main__":
    watch()
