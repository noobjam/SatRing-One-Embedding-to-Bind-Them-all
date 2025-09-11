import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Robust imports whether running as a module (-m) or as a script
try:
    from src.encoders.s1_encoder import S1_Encoder
    from src.encoders.s2_encoder import S2_Encoder
    from src.encoders.l8_9_sncoder import L8_9_Encoder
    from src.encoders.encoder_base import EncoderBase
except Exception:
    from ..encoders.s1_encoder import S1_Encoder
    from ..encoders.s2_encoder import S2_Encoder
    from ..encoders.l8_9_sncoder import L8_9_Encoder
    from ..encoders.encoder_base import EncoderBase


SUPPORTED_MODALITIES = ["S1", "S2", "L8_9"]


def info(msg: str):
    print(f"[encoder_training] {msg}", flush=True)


class AlignedMultimodalPatchDataset(Dataset):
    """
    Aligned multi-modal patches dataset.

    Expected processed data layout produced by your preprocess and patching steps:
      processed_data/
        S2/<tile_id>/patches/patch_XXXX_i_j.npy
        L8_9/<tile_id>/patches/patch_XXXX_i_j.npy
        S1/<tile_id>/patches/patch_XXXX_i_j.npy   (if available)

    Assumptions:
      - Patches across modalities are spatially aligned and share identical patch filenames under the same tile_id.
      - Channel counts per modality:
          S2:   C = 13
          L8_9: C = 11 (as per your encoder)
          S1:   C = 2
      - Patch arrays are stored as H x W x C (numpy) and will be transposed to C x H x W (torch).
    """

    def __init__(
        self,
        processed_root: str,
        modalities: List[str],
        tile_filter: Optional[List[str]] = None,
        patch_limit_per_tile: Optional[int] = None,
        patch_size: Optional[int] = None,
    ):
        self.root = Path(processed_root)
        self.modalities = modalities
        self.tile_filter = set(tile_filter) if tile_filter else None
        self.patch_limit_per_tile = patch_limit_per_tile
        self.patch_size = patch_size

        for m in self.modalities:
            if m not in SUPPORTED_MODALITIES:
                raise ValueError(f"Unsupported modality: {m}. Supported: {SUPPORTED_MODALITIES}")

        # Build aligned index by intersecting patch filenames across modalities per tile
        self.index: List[Tuple[str, str]] = self._build_index()
        info(f"AlignedMultimodalPatchDataset: {len(self.index)} aligned patches across {self.modalities}")

    def _tiles_for_modality(self, modality: str) -> List[str]:
        mdir = self.root / modality
        if not mdir.exists():
            return []
        tiles = []
        for child in mdir.iterdir():
            if child.is_dir():
                tiles.append(child.name)
        return tiles

    def _patches_for_tile_modality(self, tile_id: str, modality: str) -> List[str]:
        pdir = self.root / modality / tile_id / "patches"
        if not pdir.exists():
            return []
        return sorted([p.name for p in pdir.glob("*.npy")])

    def _build_index(self) -> List[Tuple[str, str]]:
        # Find tiles shared across all modalities
        modality_tiles = []
        for m in self.modalities:
            tiles = self._tiles_for_modality(m)
            if self.tile_filter:
                tiles = [t for t in tiles if t in self.tile_filter]
            modality_tiles.append(set(tiles))

        if not modality_tiles:
            return []

        shared_tiles = modality_tiles[0]
        for tset in modality_tiles[1:]:
            shared_tiles = shared_tiles.intersection(tset)

        if len(shared_tiles) == 0:
            info("Warning: no shared tiles across modalities. Dataset will be empty.")
            return []

        index: List[Tuple[str, str]] = []
        for tile_id in sorted(shared_tiles):
            # Intersect patch filenames across modalities
            shared_patches: Optional[set] = None
            for m in self.modalities:
                patches = set(self._patches_for_tile_modality(tile_id, m))
                if len(patches) == 0:
                    shared_patches = set() if shared_patches is None else shared_patches.intersection(set())
                    break
                shared_patches = patches if shared_patches is None else shared_patches.intersection(patches)

            if not shared_patches:
                continue

            patches_sorted = sorted(list(shared_patches))
            if self.patch_limit_per_tile:
                patches_sorted = patches_sorted[: self.patch_limit_per_tile]

            index.extend([(tile_id, p) for p in patches_sorted])

        return index

    def __len__(self) -> int:
        return len(self.index)

    def _load_patch(self, tile_id: str, patch_name: str, modality: str) -> torch.Tensor:
        f = self.root / modality / tile_id / "patches" / patch_name
        arr = np.load(f)  # H W C
        if self.patch_size is not None:
            # Optional sanity check/crop if desired
            H, W, _ = arr.shape
            if H < self.patch_size or W < self.patch_size:
                raise ValueError(f"Patch smaller than expected: {f} shape={arr.shape} required={self.patch_size}")
        # to C H W
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        ten = torch.from_numpy(arr)
        return ten

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile_id, patch_name = self.index[idx]
        sample: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            try:
                sample[m] = self._load_patch(tile_id, patch_name, m)
            except Exception:
                # Should not happen if index is aligned, but keep robust
                continue
        return sample


class MultiModalInfoNCE(nn.Module):
    """
    Multi-modal InfoNCE loss: computes NT-Xent across all available modality pairs.

    - Inputs are dict modality->embeddings (B, D), with L2-normalized vectors recommended.
    - For each pair (i, j), compute similarity matrix sim = z_i @ z_j.T / tau
      and do cross-entropy with diagonal as positives.
    - Average across pairs present.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def _pairwise_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float, ce: nn.Module) -> torch.Tensor:
        """
        z1, z2: (B, D) normalized embeddings
        """
        sim = (z1 @ z2.t()) / tau  # (B, B)
        targets = torch.arange(sim.size(0), device=sim.device)
        loss_12 = ce(sim, targets)
        loss_21 = ce(sim.t(), targets)
        return 0.5 * (loss_12 + loss_21)

    def forward(self, embeds: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = list(embeds.keys())
        n = len(keys)
        if n < 2:
            raise ValueError("Need at least two modalities to compute multi-modal InfoNCE.")

        losses = []
        for i in range(n):
            for j in range(i + 1, n):
                z1 = embeds[keys[i]]
                z2 = embeds[keys[j]]
                # ensure equal batch sizes
                b = min(z1.size(0), z2.size(0))
                if b == 0:
                    continue
                losses.append(self._pairwise_loss(z1[:b], z2[:b], self.tau, self.ce))

        if len(losses) == 0:
            raise ValueError("No valid modality pairs found in batch for loss computation.")
        return torch.stack(losses).mean()


def build_encoders(
    embed_dim: int,
    backbone: str,
    modalities: List[str],
    normalize: bool = True,
) -> Dict[str, EncoderBase]:
    encoders: Dict[str, EncoderBase] = {}
    for m in modalities:
        if m == "S2":
            encoders[m] = S2_Encoder(embed_dim=embed_dim, backbone_type=backbone)
        elif m == "L8_9":
            encoders[m] = L8_9_Encoder(embed_dim=embed_dim, backbone_type=backbone)
        elif m == "S1":
            encoders[m] = S1_Encoder(embed_dim=embed_dim, backbone_type=backbone)
        else:
            raise ValueError(f"Unsupported modality: {m}")
        # set EncoderBase.normalize flag where applicable
        if hasattr(encoders[m], "normalize"):
            encoders[m].normalize = normalize
    return encoders


def freeze_backbones(encoders: Dict[str, EncoderBase], freeze: bool):
    if not freeze:
        return
    for m, enc in encoders.items():
        for p in enc.parameters():
            p.requires_grad = False
        info(f"Froze all params of {m} encoder.")


def unfreeze_last_linear(encoders: Dict[str, EncoderBase]):
    """
    Optionally unfreeze the last linear layer for fine-tuning projection to embed_dim.
    """
    for m, enc in encoders.items():
        last_linear = None
        # Try common locations
        if hasattr(enc, "encoder"):
            # torchvision resnet head is usually at encoder.fc
            if hasattr(enc.encoder, "fc") and isinstance(enc.encoder.fc, nn.Linear):
                last_linear = enc.encoder.fc
        if last_linear is not None:
            for p in last_linear.parameters():
                p.requires_grad = True
            info(f"Unfroze last linear layer for {m}.")


def param_groups(encoders: Dict[str, EncoderBase]) -> List[dict]:
    params = []
    for m, enc in encoders.items():
        params.append({"params": [p for p in enc.parameters() if p.requires_grad], "name": f"encoder_{m}"})
    return params


def train_one_epoch(
    encoders: Dict[str, EncoderBase],
    loss_fn: MultiModalInfoNCE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modalities: List[str],
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    for enc in encoders.values():
        enc.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        # Move inputs to device and forward through encoders
        embeds: Dict[str, torch.Tensor] = {}
        for m in modalities:
            if m in batch:
                x = batch[m].to(device, non_blocking=True)  # (B, C, H, W)
                embeds[m] = encoders[m].encode(x) if hasattr(encoders[m], "encode") else encoders[m](x)
        if len(embeds) < 2:
            # skip if not enough modalities present in the batch
            continue

        loss = loss_fn(embeds)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss, total_batches


@torch.no_grad()
def evaluate(
    encoders: Dict[str, EncoderBase],
    loss_fn: MultiModalInfoNCE,
    loader: DataLoader,
    device: torch.device,
    modalities: List[str],
) -> float:
    for enc in encoders.values():
        enc.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        embeds: Dict[str, torch.Tensor] = {}
        for m in modalities:
            if m in batch:
                x = batch[m].to(device, non_blocking=True)
                embeds[m] = encoders[m].encode(x) if hasattr(encoders[m], "encode") else encoders[m](x)
        if len(embeds) < 2:
            continue
        loss = loss_fn(embeds)
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def save_checkpoint(
    encoders: Dict[str, EncoderBase],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    states = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "encoders": {m: enc.state_dict() for m, enc in encoders.items()},
    }
    f = out_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(states, f)
    info(f"Saved checkpoint: {f}")


def build_dataloaders(
    processed_root: str,
    modalities: List[str],
    batch_size: int,
    num_workers: int,
    val_split: float = 0.05,
    patch_limit_per_tile: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset = AlignedMultimodalPatchDataset(
        processed_root=processed_root,
        modalities=modalities,
        tile_filter=None,
        patch_limit_per_tile=patch_limit_per_tile,
        patch_size=patch_size,
    )
    if len(dataset) == 0:
        info("Warning: dataset is empty. Check your processed data layout and patch alignment.")
    n = len(dataset)
    n_val = int(math.floor(val_split * n)) if n > 0 else 0
    n_train = n - n_val
    if n_val > 0:
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    else:
        train_set, val_set = dataset, None

    def collate_fn(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Collate per modality
        out: Dict[str, List[torch.Tensor]] = {}
        for sample in batch_list:
            for m, x in sample.items():
                out.setdefault(m, []).append(x)
        out_stacked = {m: torch.stack(xs, dim=0) for m, xs in out.items()}
        return out_stacked

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train modality encoders to a shared latent space using InfoNCE.")
    parser.add_argument("--processed_data", type=str, default="./data/processed", help="Path to processed data root.")
    parser.add_argument("--modalities", type=str, nargs="+", default=["S2", "L8_9"], choices=SUPPORTED_MODALITIES)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before contrastive loss.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze all encoder params (for head-only training).")
    parser.add_argument("--unfreeze_last_linear", action="store_true", help="Unfreeze last linear layer when backbone is frozen.")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--patch_limit_per_tile", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Using device: {device}")

    # Data
    train_loader, val_loader = build_dataloaders(
        processed_root=args.processed_data,
        modalities=args.modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        patch_limit_per_tile=args.patch_limit_per_tile,
        patch_size=args.patch_size,
    )

    # Models
    encoders = build_encoders(
        embed_dim=args.embed_dim,
        backbone=args.backbone,
        modalities=args.modalities,
        normalize=args.normalize,
    )
    for m, enc in encoders.items():
        enc.to(device)

    if args.freeze_backbone:
        freeze_backbones(encoders, freeze=True)
        if args.unfreeze_last_linear:
            unfreeze_last_linear(encoders)

    # Optimizer
    pg = param_groups(encoders)
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)

    # Loss
    loss_fn = MultiModalInfoNCE(temperature=args.temperature)

    # Train loop
    best_val = float("inf")
    out_dir = Path(args.output_dir)
    for epoch in range(1, args.epochs + 1):
        avg_loss, num_batches = train_one_epoch(
            encoders=encoders,
            loss_fn=loss_fn,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            modalities=args.modalities,
            grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        )
        info(f"Epoch {epoch}/{args.epochs} - train loss: {avg_loss:.4f} ({num_batches} batches)")

        if val_loader is not None:
            val_loss = evaluate(
                encoders=encoders,
                loss_fn=loss_fn,
                loader=val_loader,
                device=device,
                modalities=args.modalities,
            )
            info(f"Epoch {epoch}/{args.epochs} - val loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(encoders, optimizer, epoch, out_dir)
        else:
            # Save periodically if no val set
            if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
                save_checkpoint(encoders, optimizer, epoch, out_dir)

    info("Training completed.")


if __name__ == "__main__":
    main()
