"""
Checkpoint save/load utilities for AttenX training.

Supports saving and restoring generator, discriminators,
optimizers, and schedulers. Handles dynamic numbers of
discriminators (not hardcoded to three).
"""

import torch


def save_checkpoint(path, epoch, netG, netsD, optimizerG, optimizersD,
                    schedulerG=None, schedulersD=None, best_val_loss=None):
    """
    Save a full training checkpoint to disk.

    Args:
        path: Destination file path.
        epoch: Current epoch number.
        netG: Generator model.
        netsD: List of discriminator models.
        optimizerG: Generator optimizer.
        optimizersD: List of discriminator optimizers.
        schedulerG: Optional generator LR scheduler.
        schedulersD: Optional list of discriminator LR schedulers.
        best_val_loss: Best validation loss so far.
    """
    state = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "best_val_loss": best_val_loss,
    }
    for i, netD in enumerate(netsD):
        state[f"netD{i}"] = netD.state_dict()
    for i, optD in enumerate(optimizersD):
        state[f"optimizerD{i}"] = optD.state_dict()
    if schedulerG is not None:
        state["schedulerG"] = schedulerG.state_dict()
    if schedulersD is not None:
        state["schedulersD"] = [s.state_dict() for s in schedulersD]
    torch.save(state, path)


def load_checkpoint(path, netG, netsD, optimizerG=None, optimizersD=None,
                    device="cpu"):
    """
    Load a training checkpoint and restore model/optimizer states.

    Args:
        path: Checkpoint file path.
        netG: Generator model instance.
        netsD: List of discriminator model instances.
        optimizerG: Optional generator optimizer to restore.
        optimizersD: Optional list of discriminator optimizers to restore.
        device: Device to map checkpoint tensors to.

    Returns:
        Tuple of (epoch, best_val_loss).
    """
    state = torch.load(path, map_location=device)
    netG.load_state_dict(state["netG"])
    for i, netD in enumerate(netsD):
        key = f"netD{i}"
        if key in state:
            netD.load_state_dict(state[key])
    if optimizerG is not None:
        optimizerG.load_state_dict(state["optimizerG"])
        _move_optimizer(optimizerG, device)
    if optimizersD is not None:
        for i, opt in enumerate(optimizersD):
            key = f"optimizerD{i}"
            if key in state:
                opt.load_state_dict(state[key])
                _move_optimizer(opt, device)
    return state.get("epoch", 0), state.get("best_val_loss", float("inf"))


def _move_optimizer(optimizer, device):
    """
    Move all optimizer state tensors to the specified device.

    Args:
        optimizer: PyTorch optimizer whose state tensors need moving.
        device: Target torch device.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)
