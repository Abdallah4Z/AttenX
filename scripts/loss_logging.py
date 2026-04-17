import torch
import os
import json

class LossLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = []

    def log(self, step, losses):
        """
        losses: dict of {name: value}
        """
        losses['step'] = step
        self.history.append(losses)
        
        # Log to console
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items() if k != 'step'])
        print(f"Step [{step}] | {loss_str}")

        # Warning signs logic
        if losses.get('D_Loss', 1.0) < 0.01:
            print("  ⚠️  WARNING: Discriminator loss approaching 0. Possible mode collapse or over-optimized D.")
        if losses.get('G_Loss', 0.0) > 10.0:
            print("  ⚠️  WARNING: Generator loss is unusually high.")

    def save(self, filename='training_log.json'):
        with open(os.path.join(self.log_dir, filename), 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Full log saved to {os.path.join(self.log_dir, filename)}")

if __name__ == "__main__":
    logger = LossLogger('logs/training')
    for i in range(5):
        logger.log(i * 100, {
            'D_Loss': 0.8 - (i * 0.1),
            'G_Loss': 1.5 + (i * 0.2),
            'L_Attn': 0.5,
            'L_DAMSM': 2.3
        })
    logger.save()
