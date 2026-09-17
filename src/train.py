import wandb
import hydra 
from tqdm import tqdm
from omegaconf import OmegaConf
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, all_reduce, ReduceOp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data import HFDataset
from src.models import Encoder, SIGReg

from utils import setup_ddp, get_backbone, load_checkpoint, save_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="vit_tiny8_inet10.yaml")
def train(cfg: OmegaConf):
    rank, local_rank, world_size, device = setup_ddp()

    wandb.init(project="LeJEPA", config=OmegaConf.to_container(cfg))
    
    torch.manual_seed(cfg.seed+rank)

    train_ds = HFDataset(cfg.data.dataset_name, split="train", V=cfg.data.V)
    val_ds = HFDataset(cfg.data.dataset_name, split="validation", V=1)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                       shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=False, 
                              sampler=train_sampler, num_workers=cfg.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, 
                            sampler=val_sampler, num_workers=cfg.data.num_workers, pin_memory=True)

    backbone = get_backbone(cfg.model.arch, cfg.model.size)
    encoder = Encoder(backbone, dim=cfg.model.dim, proj_dim=cfg.model.proj_dim).to(device)
    encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
    encoder = torch.compile(encoder)

    probe = nn.Sequential(
        nn.LayerNorm(cfg.model.dim),
        nn.Linear(cfg.model.dim, cfg.data.n_classes)
    ).to(device)
    probe = DDP(probe, device_ids=[local_rank], output_device=local_rank)

    sigreg = SIGReg(cfg.model.n_proj).to(device)
    sigreg = DDP(sigreg, device_ids=[local_rank], output_device=local_rank)

    g1 = {"params": encoder.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    optimizer = optim.AdamW([g1, g2])

    total_steps = len(train_loader) * cfg.epochs
    s1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=len(train_loader))
    s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-len(train_loader), eta_min=1e-3)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[len(train_loader)])

    scaler = torch.GradScaler("cuda")

    if cfg.ckpt_path is not None:
        start_epoch = load_checkpoint(
            cfg.resume_ckpt, encoder, probe, optimizer, scheduler, scaler, device
        )

        if rank == 0:
            print(f"Resumed training from epoch {start_epoch} using checkpoint {cfg.resume_ckpt}")

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        encoder.train(), probe.train()

        for xs, ys in tqdm(train_loader, desc=f"Training epoch [{epoch}/{cfg.epochs}]", disable=(rank!=0)):
            xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                embs, projs = encoder(xs)
                inv_loss = (projs.mean(0) - projs).square().mean()
                sigreg_loss = sigreg(projs)
                lejepa_loss = (1 - cfg.lamd) * inv_loss + cfg.lamd * sigreg_loss
                y = ys.repeat_interleave(cfg.data.V)
                probe_loss = F.cross_entropy(probe(embs.detach()), y)
                loss = lejepa_loss + probe_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0:
                wandb.log({
                    "train/inv_loss": inv_loss.item(),
                    "train/sigreg_loss": sigreg_loss.item(),
                    "train/lejepa_loss": lejepa_loss.item(),
                    "train/probe_loss": probe_loss.item()
                })

        val_sampler.set_epoch(epoch)
        encoder.eval(), probe.eval()
        correct = torch.tensor(0, device=device, dtype=torch.int64)
        with torch.inference_mode():
            for xs, ys in tqdm(val_loader, desc=f"Validation epoch [{epoch}/{cfg.epochs}]"):
                xs, ys = xs.to(device), ys.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    embs, _ = encoder(xs)
                    correct += (probe(embs).argmax(1), ys).sum().item()

        all_reduce(correct, op=ReduceOp.SUM)

        if rank == 0:
            wandb.log({"val/accuracy": correct/len(val_ds), "val/epoch": epoch})

        if rank == 0 and ((epoch+1) % cfg.ckpt_every or (epoch+1) == cfg.epochs):
            save_checkpoint(cfg.ckpt_dir, encoder, probe, scaler, optimizer, scheduler, epoch)

    if rank == 0:
        wandb.finish()
        
    destroy_process_group()


if __name__ == "__main__":
    train()
