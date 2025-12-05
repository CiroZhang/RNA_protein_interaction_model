#!/usr/bin/env python3
"""Quick script to evaluate the random baseline model."""

import torch
import numpy as np
from tqdm import tqdm
from sacrebleu import BLEU
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

from dataset import read_data
from genechat_model import RandomSummaryBaseline

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 80


def compute_metrics(ref, pred):
    """Compute BLEU-n, METEOR, ROUGE-1, ROUGE-L."""
    # BLEU scorers
    bleu1_scorer = BLEU(max_ngram_order=1, effective_order=True)
    bleu2_scorer = BLEU(max_ngram_order=2, effective_order=True)
    bleu3_scorer = BLEU(max_ngram_order=3, effective_order=True)
    bleu4_scorer = BLEU(max_ngram_order=4, effective_order=True)

    # ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # BLEU scores
    b1 = bleu1_scorer.sentence_score(pred, [ref]).score / 100
    b2 = bleu2_scorer.sentence_score(pred, [ref]).score / 100
    b3 = bleu3_scorer.sentence_score(pred, [ref]).score / 100
    b4 = bleu4_scorer.sentence_score(pred, [ref]).score / 100

    # METEOR
    ref_tok = ref.split()
    pred_tok = pred.split()
    meteor = meteor_score([ref_tok], pred_tok)

    # ROUGE
    r = rouge.score(ref, pred)
    rouge1 = r["rouge1"].fmeasure
    rougeL = r["rougeL"].fmeasure

    return {
        "bleu1": b1,
        "bleu2": b2,
        "bleu3": b3,
        "bleu4": b4,
        "meteor": meteor,
        "rouge1": rouge1,
        "rougeL": rougeL,
    }


def main():
    print("Loading data...")
    train_data, test_data, _, _ = read_data()

    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")

    print("\nCreating random baseline model...")
    model = RandomSummaryBaseline(train_data).to(DEVICE)

    print("Evaluating random baseline...")
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    meteor_scores = []
    rouge1_scores, rougeL_scores = [], []

    for ex in tqdm(test_data, desc="Evaluating random"):
        dna = ex["dna"]
        ref = ex["target"]

        pred = model.generate(dna=dna, max_new_tokens=MAX_NEW_TOKENS, device=DEVICE)

        m = compute_metrics(ref, pred)

        bleu1_scores.append(m["bleu1"])
        bleu2_scores.append(m["bleu2"])
        bleu3_scores.append(m["bleu3"])
        bleu4_scores.append(m["bleu4"])
        meteor_scores.append(m["meteor"])
        rouge1_scores.append(m["rouge1"])
        rougeL_scores.append(m["rougeL"])

    # Print results
    print("\n" + "="*80)
    print("RANDOM BASELINE RESULTS")
    print("="*80)
    print(f"BLEU-1:  {np.mean(bleu1_scores):.6f}")
    print(f"BLEU-2:  {np.mean(bleu2_scores):.6f}")
    print(f"BLEU-3:  {np.mean(bleu3_scores):.6f}")
    print(f"BLEU-4:  {np.mean(bleu4_scores):.6f}")
    print(f"METEOR:  {np.mean(meteor_scores):.6f}")
    print(f"ROUGE-1: {np.mean(rouge1_scores):.6f}")
    print(f"ROUGE-L: {np.mean(rougeL_scores):.6f}")
    print("="*80)

    # Save the metrics
    metrics = {
        'bleu1': bleu1_scores,
        'bleu2': bleu2_scores,
        'bleu3': bleu3_scores,
        'bleu4': bleu4_scores,
        'meteor': meteor_scores,
        'rouge1': rouge1_scores,
        'rougeL': rougeL_scores,
    }

    torch.save(metrics, 'random_metrics.pt')
    print("\n✓ Saved metrics to random_metrics.pt")


if __name__ == "__main__":
    main()
