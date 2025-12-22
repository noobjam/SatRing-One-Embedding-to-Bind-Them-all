import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from seco_model import SeCoModel
from sequenceDataset import SequenceDataset
from trainer.seco_loss import seco_loss

def ddp_setup():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://')
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--sequence-length", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Starting SeCo Training on {device}")

    # Dataset
    dataset = SequenceDataset(
        processed_data=args.data_dir,
        sequence_length=args.sequence_length,
        augment=True
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Model
    model = SeCoModel().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training Loop
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(loader):
            # Unpack batch
            v1_img = batch['v1_img'].to(device)
            v1_t = batch['v1_t'].to(device)
            v1_s = batch['v1_s'].to(device)
            v1_osm = batch['v1_osm'].to(device)  # NEW: OSM tags
            
            v2_img = batch['v2_img'].to(device)
            v2_t = batch['v2_t'].to(device)
            v2_s = batch['v2_s'].to(device)
            v2_osm = batch['v2_osm'].to(device)  # NEW: OSM tags
            
            # Forward
            z0_1, z1_1, z2_1, z3_1 = model(v1_img, v1_t, v1_s)
            z0_2, z1_2, z2_2, z3_2 = model(v2_img, v2_t, v2_s)
            
            # Loss (4 subspaces now)
            l_all, l_season, l_sensor, l_semantic = seco_loss(
                z0_1, z0_2, z1_1, z1_2, z2_1, z2_2, z3_1, z3_2,
                v1_t, v2_t, v1_s, v2_s, v1_osm, v2_osm
            )
            
            # Weighted sum (can be tuned)
            loss = l_all + l_season + l_sensor + l_semantic
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch} Step {batch_idx} Loss: {loss.item():.4f} "
                      f"(All: {l_all:.4f}, Season: {l_season:.4f}, Sensor: {l_sensor:.4f}, Semantic: {l_semantic:.4f})")
        
        if rank == 0:
            print(f"Epoch {epoch} Complete. Avg Loss: {epoch_loss / len(loader):.4f}")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, Path(args.save_dir) / f"seco_epoch_{epoch}.pt", rank)

    cleanup()

if __name__ == "__main__":
    main()
