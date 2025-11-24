# main.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

from dataset import read_data, GeneSummaryDataset, collate_fn
from genechat_model import GeneChatModel

# ---------------------------------------------------------
# Device setup
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"

print(f"\n>>> Using device: {device}  (AMP: {use_amp})\n")

print(">>> Loading data...")
train_data, test_data, _, _ = read_data()
print(f"Total train entries: {len(train_data)}")

train, val = train_test_split(train_data, test_size=0.05, random_state=42)
print(f"Train split: {len(train)}   Val split: {len(val)}\n")

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
print(">>> Building model (DNABERT + GPT-2)...")
model = GeneChatModel(
    gene_chunk_nt=512,
    gene_chunk_overlap=0,
    freeze_gene_encoder=True,
).to(device)
print("Model built.\n")

# ---------------------------------------------------------
# Freeze / unfreeze GPT-2 parts
#   - Freeze all GPT-2 by default
#   - Unfreeze last 6 transformer blocks + lm_head
#   - Always train gene_adaptor
# ---------------------------------------------------------
for p in model.lm.parameters():
    p.requires_grad = False

# unfreeze last 6 transformer blocks (if there are that many)
num_blocks = len(model.lm.transformer.h)
unfreeze_k = min(6, num_blocks)
top_blocks = list(model.lm.transformer.h[-unfreeze_k:])
for blk in top_blocks:
    for p in blk.parameters():
        p.requires_grad = True

# unfreeze lm_head
for p in model.lm.lm_head.parameters():
    p.requires_grad = True

# gene_adaptor always trainable
for p in model.gene_adaptor.parameters():
    p.requires_grad = True

# ---------------------------------------------------------
# Dataset + Dataloaders
# ---------------------------------------------------------
train_ds = GeneSummaryDataset(train)
val_ds   = GeneSummaryDataset(val)

num_workers = 8
pin_mem = device == "cuda"

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_mem,
)
val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_mem,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}\n")

# ---------------------------------------------------------
# Optimizer + AMP scaler
#   - Higher LR for adapter
#   - Lower LR for GPT-2 top layers
# ---------------------------------------------------------
adaptor_params = list(model.gene_adaptor.parameters())

gpt2_top_params = []
for blk in top_blocks:
    gpt2_top_params.extend(list(blk.parameters()))
gpt2_top_params.extend(list(model.lm.lm_head.parameters()))

optimizer = torch.optim.AdamW(
    [
        {"params": adaptor_params, "lr": 1e-4},
        {"params": gpt2_top_params, "lr": 5e-5},
    ]
)

scaler = GradScaler("cuda") if use_amp else None

# ---------------------------------------------------------
# Training / eval functions
# ---------------------------------------------------------
def run_one_epoch(loader, train: bool = True) -> float:
    if train:
        model.train()
        desc = "Training"
    else:
        model.eval()
        desc = "Validating"

    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, ncols=110)

    for dna_list, tgt_list in pbar:
        batch_loss = 0.0
        B = len(dna_list)

        if train:
            optimizer.zero_grad(set_to_none=True)

        for dna, target in zip(dna_list, tgt_list):
            ctx = autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with ctx:
                loss = model.forward_single(dna, target, device=device)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / B

        if train:
            if use_amp:
                scaler.scale(batch_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += batch_loss.item()
        avg_loss = total_loss / (pbar.n + 1)
        pbar.set_postfix({"batch_loss": batch_loss.item(), "avg_loss": f"{avg_loss:.4f}"})

    return total_loss / len(loader)

# ---------------------------------------------------------
# Training Loop with early stopping
# ---------------------------------------------------------
best_val = float("inf")
best_path = "genechat_checkpoints/model_best.pt"
final_path = "genechat_checkpoints/model_final.pt"
os.makedirs("genechat_checkpoints", exist_ok=True)

patience = 3
no_improve = 0
max_epochs = 12  # more epochs now that we partially unfreeze GPT-2

print("\n====================")
print("  START TRAINING")
print("====================\n")

for epoch in range(1, max_epochs + 1):
    print(f"\n===== Epoch {epoch} =====")

    train_loss = run_one_epoch(train_loader, train=True)
    val_loss   = run_one_epoch(val_loader, train=False)

    print(f"\nEpoch {epoch} summary:")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val   loss: {val_loss:.4f}")

    if val_loss < best_val - 1e-3:
        best_val = val_loss
        no_improve = 0
        torch.save(model.state_dict(), best_path)
        print(f">>> Saved BEST model to {best_path}")
    else:
        no_improve += 1
        print(f">>> No improvement for {no_improve} epoch(s).")

    if no_improve >= patience:
        print("\n>>> Early stopping triggered.")
        break

torch.save(model.state_dict(), final_path)
print(f"\n>>> Saved FINAL model to {final_path}\n")

