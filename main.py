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
from genechat_model import (
    GeneChatModel,
    DNABERTBartDecoder,
    DNABERTT5Decoder,
    DNABERTGRUDecoder,
    RandomSummaryBaseline,
)

###########################################################################
# TRAIN ALL MODELS
###########################################################################
MODELS_TO_TRAIN = ["bart", "t5", "gru", "random"]
CHECKPOINTS_DIR = "model_weights"  # All model weights will be saved here
###########################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"

print(f"\n{'='*80}")
print(f"TRAINING ALL MODELS: {MODELS_TO_TRAIN}")
print(f"Device: {device}  |  AMP: {use_amp}  |  Checkpoints: {CHECKPOINTS_DIR}")
print(f"{'='*80}\n")

print(">>> Loading data...")
train_data, test_data, _, _ = read_data()
print(f"Total train entries: {len(train_data)}")

train, val = train_test_split(train_data, test_size=0.05, random_state=42)
print(f"Train split: {len(train)}   Val split: {len(val)}\n")

# Create main checkpoints directory
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

###########################################################################
# LOOP THROUGH ALL MODELS
###########################################################################
for MODEL_NAME in MODELS_TO_TRAIN:
    print(f"\n{'#'*80}")
    print(f"#  STARTING MODEL: {MODEL_NAME.upper()}")
    print(f"{'#'*80}\n")

    ###########################################################################
    # Build selected model
    ###########################################################################
    print(f">>> Building model: {MODEL_NAME}")

    if MODEL_NAME == "genechat":
        model = GeneChatModel(
            gene_chunk_nt=512,
            gene_chunk_overlap=0,
            freeze_gene_encoder=True,
        ).to(device)

    elif MODEL_NAME == "bart":
        model = DNABERTBartDecoder(
            gene_chunk_nt=512,
            gene_chunk_overlap=0,
            freeze_gene_encoder=True,
        ).to(device)

    elif MODEL_NAME == "t5":
        model = DNABERTT5Decoder(
            gene_chunk_nt=512,
            gene_chunk_overlap=0,
            freeze_gene_encoder=True,
        ).to(device)

    elif MODEL_NAME == "gru":
        model = DNABERTGRUDecoder(
            gene_chunk_nt=512,
            gene_chunk_overlap=0,
            freeze_gene_encoder=True,
        ).to(device)

    elif MODEL_NAME == "random":
        model = RandomSummaryBaseline(train_data).to(device)

    else:
        raise ValueError("Invalid MODEL_NAME")

    print(">>> Model built.\n")


    ###########################################################################
    # PARAMETER FREEZING / OPTIMIZATION SETUP
    ###########################################################################

    if MODEL_NAME == "genechat":

        print(">>> Configuring GPT-2 freeze/unfreeze...")

        # freeze GPT-2 entire model
        for p in model.lm.parameters():
            p.requires_grad = False

        # unfreeze last 6 blocks
        num_blocks = len(model.lm.transformer.h)
        unfreeze_k = min(6, num_blocks)
        top_blocks = list(model.lm.transformer.h[-unfreeze_k:])
        for blk in top_blocks:
            for p in blk.parameters():
                p.requires_grad = True

        # unfreeze lm_head
        for p in model.lm.lm_head.parameters():
            p.requires_grad = True

        # gene adapter always trainable
        for p in model.gene_adaptor.parameters():
            p.requires_grad = True

        # optimizer setup
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

    elif MODEL_NAME in ["bart", "t5"]:

        print(">>> Using seq2seq model training.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    elif MODEL_NAME == "gru":

        print(">>> Using custom GRU decoder training.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    elif MODEL_NAME == "random":

        print(">>> Random summary model: no training needed.")
        # Dummy optimizer to keep loop clean
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    else:
        raise ValueError("Invalid MODEL_NAME")


    ###########################################################################
    # Dataset + loaders
    ###########################################################################

    train_ds = GeneSummaryDataset(train)
    val_ds   = GeneSummaryDataset(val)

    train_loader = DataLoader(
        train_ds,
        batch_size=1 if MODEL_NAME in ["bart","t5"] else 4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device=="cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1 if MODEL_NAME in ["bart","t5"] else 4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device=="cuda"),
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}\n")


    ###########################################################################
    # Training routines
    ###########################################################################

    scaler = GradScaler("cuda") if use_amp else None

    def run_one_epoch(loader, train=True):
        if train:
            model.train()
            desc = "Training"
        else:
            model.eval()
            desc = "Validating"

        total_loss = 0.0
        pbar = tqdm(loader, desc=f"{desc} ({MODEL_NAME})", ncols=120)

        for dna_list, tgt_list in pbar:
            batch_loss = 0.0
            B = len(dna_list)

            if train:
                optimizer.zero_grad(set_to_none=True)

            # loop through items since all baselines use .forward_single(dna, target)
            for dna, target in zip(dna_list, tgt_list):
                ctx = autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
                with ctx:
                    loss = model.forward_single(dna, target, device=device)
                batch_loss += loss

            batch_loss /= B

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

            pbar.set_postfix({"batch": batch_loss.item(), "avg": f"{avg_loss:.4f}"})

        return total_loss / len(loader)


    ###########################################################################
    # Training loop
    ###########################################################################

    # file naming - all models save to same directory
    best_path = f"{CHECKPOINTS_DIR}/{MODEL_NAME}_best.pt"
    final_path = f"{CHECKPOINTS_DIR}/{MODEL_NAME}_final.pt"

    best_val = float("inf")
    patience = 3
    no_improve = 0
    max_epochs = 8 if MODEL_NAME != "genechat" else 12

    print("\n====================")
    print(f"  START TRAINING: {MODEL_NAME}")
    print("====================\n")

    # random baseline does not need training — skip to next model
    if MODEL_NAME == "random":
        print("Random baseline: training skipped.\n")
        torch.save(model.state_dict(), final_path)
        print(f"Saved: {final_path}\n")
        continue

    for epoch in range(1, max_epochs + 1):
        print(f"\n===== Epoch {epoch} =====")

        train_loss = run_one_epoch(train_loader, train=True)
        val_loss   = run_one_epoch(val_loader, train=False)

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val   loss: {val_loss:.4f}")

        # Early stopping logic
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

print(f"\n{'='*80}")
print(f"ALL MODELS TRAINING COMPLETE!")
print(f"Checkpoints saved to: {CHECKPOINTS_DIR}/")
print(f"{'='*80}\n")

