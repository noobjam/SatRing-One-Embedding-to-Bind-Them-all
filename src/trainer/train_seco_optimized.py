"""
Optimized SeCo Training Script for 6x H100 GPUs

Optimizations:
- BF16 mixed precision (H100 native)
- Larger batch sizes (32-48 per GPU)
- More data workers (16 total)
- Persistent workers
- Gradient accumulation support
- Learning rate scheduling
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from seco_model import SeCoModel
from sequenceDataset import SequenceDataset
from trainer.seco_loss import seco_loss


def ddp_setup():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # Fallback for single GPU/CPU debugging
        return 0, 1, 0


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(state, path: Path, rank):
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--batch-size", type=int, default=32, help="Per-GPU batch size (default: 32)"
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--num-workers", type=int, default=16, help="Total data workers (default: 16)"
    )
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--sequence-length", type=int, default=5)
    p.add_argument(
        "--accumulation-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    p.add_argument(
        "--use-amp", action="store_true", default=True, help="Use BF16 mixed precision"
    )
    p.add_argument("--no-amp", action="store_false", dest="use_amp")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("=" * 80)
        print("SeCo Training - Optimized for H100")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(
            f"Effective batch size: {args.batch_size * world_size * args.accumulation_steps}"
        )
        print(f"Mixed precision: {'BF16' if args.use_amp else 'FP32'}")
        print(f"Data workers: {args.num_workers}")
        print(f"Gradient accumulation: {args.accumulation_steps} steps")
        print("=" * 80)

    # Dataset
    dataset = SequenceDataset(
        processed_data=args.data_dir, sequence_length=args.sequence_length, augment=True
    )

    if rank == 0:
        print(f"\nDataset: {len(dataset)} sequences")

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=4,  # Prefetch 4 batches ahead
    )

    # Model
    model = SeCoModel().to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = len(loader) // args.accumulation_steps * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    if rank == 0:
        print(f"Total training steps: {total_steps}")
        print(f"Starting training...\n")

    # Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            # Unpack batch
            v1_img = batch["v1_img"].to(device)
            v1_t = batch["v1_t"].to(device)
            v1_s = batch["v1_s"].to(device)
            v2_img = batch["v2_img"].to(device)
            v2_t = batch["v2_t"].to(device)
            v2_s = batch["v2_s"].to(device)

            # Forward with mixed precision
            with autocast(dtype=torch.bfloat16, enabled=args.use_amp):
                z0_1, z1_1, z2_1 = model(v1_img, v1_t, v1_s)
                z0_2, z1_2, z2_2 = model(v2_img, v2_t, v2_s)

                # Loss (3 subspaces)
                l_all, l_season, l_sensor = seco_loss(
                    z0_1, z0_2, z1_1, z1_2, z2_1, z2_2, v1_t, v2_t, v1_s, v2_s
                )

                # Weighted sum
                loss = l_all + l_season + l_sensor

                # Scale for gradient accumulation
                loss = loss / args.accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * args.accumulation_steps

            # Logging
            if rank == 0 and batch_idx % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch:3d} | Step {batch_idx:4d}/{len(loader)} | "
                    f"Loss: {loss.item()*args.accumulation_steps:.4f} | "
                    f"All: {l_all.item():.3f} Season: {l_season.item():.3f} "
                    f"Sensor: {l_sensor.item():.3f} | "
                    f"LR: {lr:.2e}"
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(loader)
        samples_per_sec = len(dataset) / epoch_time

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Complete")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Throughput: {samples_per_sec:.1f} samples/sec")
            print(f"{'='*80}\n")

            # Save checkpoint
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": (
                        model.module.state_dict()
                        if world_size > 1
                        else model.state_dict()
                    ),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                },
                Path(args.save_dir) / f"seco_epoch_{epoch}.pt",
                rank,
            )

    if rank == 0:
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)

    cleanup()


if __name__ == "__main__":
    main()
