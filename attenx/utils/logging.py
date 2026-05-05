import json
import os
from datetime import datetime


class LossLogger:
    """Logs scalar metrics during training and optionally
    saves to a JSON file for later analysis."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = []

    def log(self, step, metrics):
        entry = {"step": step, **metrics}
        self.history.append(entry)

    def save(self, filename=None):
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_log_{ts}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Loss log saved to {path}")
