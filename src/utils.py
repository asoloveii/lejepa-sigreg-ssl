import os
import torch
from torch.distributed import init_process_group

from models import *


def setup_ddp():
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def load_checkpoint(ckpt_path, encoder, probe, optimizer, scheduler, scaler, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at '{ckpt_path}'")

    checkpoint = torch.load(ckpt_path, map_location=device)

    encoder.module._orig_mod.load_state_dict(checkpoint["encoder"])
    probe.module._orig_mod.load_state_dict(checkpoint["probe"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint["epoch"] + 1


def save_checkpoint(ckpt_dir, encoder, probe, optimizer, scheduler, scaler, epoch):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt")

    state = {
        "epoch": epoch,
        "encoder": encoder.module._orig_mod.state_dict(),
        "probe": probe.module._orig_mod.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(state, ckpt_path)


def get_backbone(arch, size, patch_size=8):
    match [arch, size]:
        case "vit", "tiny": return vit_tiny(patch_size)
        case "vit", "small": return vit_small(patch_size)
        case "vit", "base": return vit_base(patch_size)

        case "convnext", "atto": return convnextv2_atto()
        case "convnext", "femto": return convnextv2_femto()
        case "convnext", "pico": return convnext_pico()
        case "convnext", "nano": return convnextv2_nano()
        case "convnext", "tiny": return convnextv2_tiny()
        case "convnext", "base": return convnextv2_base()

        case _:
            raise ValueError("Architecture or size not recognized: ", arch, size)
