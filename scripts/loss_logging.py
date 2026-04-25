import torch
import os
import json

class LossLogger:
    def __init__(self, log_dir, g_warn_threshold=25.0, d_warn_threshold=0.05,
                 warmup_steps=300, sustained_steps=20):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = []
        self.g_warn_threshold = g_warn_threshold
        self.d_warn_threshold = d_warn_threshold
        self.warmup_steps = warmup_steps
        self.sustained_steps = sustained_steps
        self.high_g_streak = 0

    def log(self, step, losses):
        """
        losses: dict of {name: value}
        """
        losses['step'] = step
        self.history.append(losses)
        
        # Log to console
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items() if k != 'step'])
        print(f"Step [{step}] | {loss_str}")

        # Warning signs logic (avoid noisy alerts during early warmup)
        if step < self.warmup_steps:
            return

        d_loss = losses.get('D_Loss')
        g_proxy = losses.get('G_GAN', losses.get('G_Loss'))

        if d_loss is not None and d_loss < self.d_warn_threshold:
            print("  WARNING: D_Loss is very low; discriminator may be overpowering generator.")

        if g_proxy is None:
            return

        if g_proxy > self.g_warn_threshold:
            self.high_g_streak += 1
        else:
            self.high_g_streak = 0

        if self.high_g_streak >= self.sustained_steps:
            print("  WARNING: G_GAN has stayed high for many steps; training may be imbalanced.")
            self.high_g_streak = 0

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
