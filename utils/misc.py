from math import inf
import torch as t
from torch.nn.parallel import DataParallel, DistributedDataParallel

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, step=None, loss=None):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "loss": loss
    }
    t.save(ckpt, path)

def load_checkpoint(path, model, device="cuda:0", optimizer=None, scheduler=None):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    ckpt = t.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        if ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        if ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt["epoch"] if "epoch" in ckpt else 0
    step = ckpt["step"] if "step" in ckpt else 0
    loss = ckpt["loss"] if "loss" in ckpt else float("inf")
    return epoch, step, loss
