import torch


def save_checkpoint(path, epoch, netG, netsD, optimizerG, optimizersD,
                    schedulerG=None, schedulersD=None, best_val_loss=None):
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
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)
