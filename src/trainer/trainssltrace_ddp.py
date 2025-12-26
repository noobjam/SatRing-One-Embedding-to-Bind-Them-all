import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from contrastive_loss import info_nce_loss
from projection_head import ProjectionHead
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models import TRACE
from sequenceDataset import SequenceDataset


def get_pooled_feat(pixel_emb):
    if pixel_emb.dim() == 4:
        return pixel_emb.view(pixel_emb.size(0), pixel_emb.size(1), -1).mean(dim=2)
    elif pixel_emb.dim() == 5:
        B, T, C, H, W = pixel_emb.shape
        return pixel_emb.view(B, T, C, -1).mean(dim=3)
    else:
        raise ValueError(f"Unsupported pixel_emb.dim() = {pixel_emb.dim()}")


def ddp_setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def save_on_rank0(state, path: Path):
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sensor-type", default="fusion")
    p.add_argument("--sequence-length", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(
            f"Starting DDP training: world_size={world_size}, local_rank={local_rank}, device={device}"
        )
        print("Args:", args)

    dataset = SequenceDataset(
        processed_data=args.data_dir,
        sensor_type=args.sensor_type,
        sequence_length=args.sequence_length,
        augment=True,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    backbone = TRACE(
        dim=1024, num_heads=8, temporal_depth=2, H=14, W=14, aggregate="mean"
    ).to(device)
    projector = ProjectionHead(in_dim=1024, hidden_dim=512, out_dim=args.proj_dim).to(
        device
    )
    backbone = DDP(
        backbone,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    projector = DDP(projector, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_bf16 = args.bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler() if not use_bf16 else None

    save_dir = Path(args.save_dir)
    start_time = time.time()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        backbone.train()
        projector.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        iters = 0

        for batch_idx, batch in enumerate(loader):
            anchor_seq = batch["v1_img"].to(device, non_blocking=True)
            pair_seq = batch["v2_img"].to(device, non_blocking=True)
            # Assuming loc_idx is used for something or we create a dummy label for now
            # In SSL, usually all samples in a batch are positive views of themselves.
            label = torch.ones(anchor_seq.shape[0], device=device)

            B = anchor_seq.shape[0]
            T = anchor_seq.shape[1] if anchor_seq.dim() == 5 else 1
            timesteps = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)

            # --- Debug print for first batch
            if rank == 0 and epoch == 0 and batch_idx == 0:
                print("=== Debug shapes for first batch ===")
                print(
                    f"anchor_seq: {anchor_seq.shape}, pair_seq: {pair_seq.shape}, label: {label.shape}"
                )

            with torch.cuda.amp.autocast(enabled=True, dtype=autocast_dtype):
                anchor_pixel = backbone(anchor_seq, timesteps)
                pair_pixel = backbone(pair_seq, timesteps)
                anchor_feat = get_pooled_feat(anchor_pixel)
                pair_feat = get_pooled_feat(pair_pixel)

                if rank == 0 and epoch == 0 and batch_idx == 0:
                    print(
                        f"anchor_pixel: {anchor_pixel.shape}, pair_pixel: {pair_pixel.shape}"
                    )
                    print(
                        f"anchor_feat: {anchor_feat.shape}, pair_feat: {pair_feat.shape}"
                    )

                z_a = projector(anchor_feat)
                z_b = projector(pair_feat)
                mask_pos = label == 1

                if rank == 0 and epoch == 0 and batch_idx == 0:
                    print(
                        f"z_a: {z_a.shape}, z_b: {z_b.shape}, mask_pos: {mask_pos.shape}, positive_count: {mask_pos.sum().item()}"
                    )
                    print("=== End debug shapes ===")

                if mask_pos.sum() == 0 or (~mask_pos).sum() == 0:
                    continue

                loss = (
                    info_nce_loss(
                        z_a[mask_pos], z_b[mask_pos], temperature=args.temperature
                    )
                    / args.accum_steps
                )

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.accum_steps
            iters += 1
            global_step += 1

            if global_step % 50 == 0 and rank == 0:
                avg_loss_now = epoch_loss / max(1, iters)
                elapsed = time.time() - start_time
                print(
                    f"[Rank {rank}] Epoch {epoch+1} Step {global_step} Batch {batch_idx+1}/{len(loader)} "
                    f"Loss {loss.item() * args.accum_steps:.4f} Avg {avg_loss_now:.4f} Time {elapsed:.1f}s"
                )

        if iters == 0:
            reduced_avg_loss = float("inf")
        else:
            local_avg = torch.tensor(
                epoch_loss / max(1, iters), device=device, dtype=torch.float32
            )
            dist.all_reduce(local_avg, op=dist.ReduceOp.SUM)
            reduced_avg_loss = local_avg.item() / world_size

        if rank == 0:
            print(
                f"Epoch {epoch+1} finished. Avg loss (all ranks): {reduced_avg_loss:.4f}"
            )
            ckpt = {
                "epoch": epoch + 1,
                "backbone_state": backbone.module.state_dict(),
                "projector_state": projector.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "avg_loss": reduced_avg_loss,
                "args": vars(args),
            }
            save_on_rank0(ckpt, save_dir / f"trace_ddp_epoch{epoch+1}.pt")
            if reduced_avg_loss < best_loss:
                best_loss = reduced_avg_loss
                save_on_rank0(ckpt, save_dir / "trace_ddp_best.pt")

    cleanup()
    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
