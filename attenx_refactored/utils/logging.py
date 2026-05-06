"""
Training loss logger.

Records scalar metrics per training step and optionally
persists them to a JSON file for later analysis.
"""

import json
import os
from datetime import datetime


class LossLogger:
    """
    Accumulates training metrics and saves to JSON.

    Usage:
        logger = LossLogger("./logs")
        logger.log(step, {"G_Loss": 0.5, "L_KL": 0.1})
        logger.save()
    """

    def __init__(self, log_dir):
        """
        Initialize the logger.

        Args:
            log_dir: Directory path for saving log files.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = []

    def log(self, step, metrics):
        """
        Record a set of metrics at a given training step.

        Args:
            step: Current global training step.
            metrics: Dictionary of metric names to scalar values.
        """
        entry = {"step": step, **metrics}
        self.history.append(entry)

    def save(self, filename=None):
        """
        Save the accumulated log history to a JSON file.

        Args:
            filename: Optional custom filename; defaults to
                      loss_log_{timestamp}.json.
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_log_{ts}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Loss log saved to {path}")
