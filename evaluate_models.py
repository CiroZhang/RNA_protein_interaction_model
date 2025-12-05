#!/usr/bin/env python3
"""
Model Evaluation Script
Measures ROUGE-1, ROUGE-L, BLEU-1, BLEU-2, BLEU-3, BLEU-4, and METEOR
for all trained models and generates comparison visualizations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sacrebleu import BLEU
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import pandas as pd

from dataset import read_data
from genechat_model import (
    GeneChatModel,
    DNABERTBartDecoder,
    DNABERTT5Decoder,
    DNABERTGRUDecoder,
    RandomSummaryBaseline,
)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS_DIR = "model_weights"
GENE_CHUNK_NT = 512
GENE_CHUNK_OVERLAP = 0
FREEZE_GENE_ENCODER = True
MAX_NEW_TOKENS = 80

print(f"\n{'='*80}")
print("MODEL EVALUATION SCRIPT")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Model weights directory: {MODEL_WEIGHTS_DIR}")
print(f"{'='*80}\n")


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


def load_model(model_name, model_class, checkpoint_path):
    """Load a model from checkpoint."""
    print(f"Loading {model_name} from {checkpoint_path}...")

    if model_name == "random":
        # Random baseline needs training data
        train_data, _, _, _ = read_data()
        model = model_class(train_data).to(DEVICE)
    else:
        model = model_class(
            gene_chunk_nt=GENE_CHUNK_NT,
            gene_chunk_overlap=GENE_CHUNK_OVERLAP,
            freeze_gene_encoder=FREEZE_GENE_ENCODER,
        ).to(DEVICE)

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    print(f"✓ {model_name} loaded successfully")
    return model


def evaluate_model(model, test_data, model_name):
    """Evaluate a single model on test data."""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*80}")

    # Storage lists
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    meteor_scores = []
    rouge1_scores, rougeL_scores = [], []

    # Evaluate on test set
    for ex in tqdm(test_data, desc=f"Evaluating {model_name}"):
        dna = ex["dna"]
        ref = ex["target"]

        pred = model.generate(
            dna=dna,
            max_new_tokens=MAX_NEW_TOKENS,
            device=DEVICE,
        )

        m = compute_metrics(ref, pred)

        bleu1_scores.append(m["bleu1"])
        bleu2_scores.append(m["bleu2"])
        bleu3_scores.append(m["bleu3"])
        bleu4_scores.append(m["bleu4"])
        meteor_scores.append(m["meteor"])
        rouge1_scores.append(m["rouge1"])
        rougeL_scores.append(m["rougeL"])

    return {
        'bleu1': bleu1_scores,
        'bleu2': bleu2_scores,
        'bleu3': bleu3_scores,
        'bleu4': bleu4_scores,
        'meteor': meteor_scores,
        'rouge1': rouge1_scores,
        'rougeL': rougeL_scores,
    }


def generate_comprehensive_report(results_summary, all_model_metrics, output_path='evaluation_report.png'):
    """Generate a comprehensive one-page report with all visualizations."""
    print(f"\nGenerating comprehensive evaluation report...")

    num_models = len(all_model_metrics)

    # Calculate figure size based on number of models
    # Top section: comparison bars (2 rows x 4 cols = 8 subplots)
    # Bottom section: distributions for each model (7 plots per model, arranged in rows)
    fig = plt.figure(figsize=(24, 8 + num_models * 6))

    # Use GridSpec for better layout control
    gs = gridspec.GridSpec(2 + num_models, 1, height_ratios=[4] + [6] * num_models + [0.5],
                          hspace=0.4, figure=fig)

    # Add main title
    fig.suptitle('Model Evaluation Report - Comprehensive Metrics Analysis',
                 fontsize=24, fontweight='bold', y=0.995)

    # ========== SECTION 1: COMPARISON BAR CHARTS ==========
    gs_bars = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0],
                                                hspace=0.35, wspace=0.3)

    metrics_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-1', 'ROUGE-L']
    model_names = list(results_summary.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, metric_name in enumerate(metrics_names):
        ax = fig.add_subplot(gs_bars[idx // 4, idx % 4])
        values = [results_summary[m][metric_name] for m in model_names]
        bars = ax.bar(model_names, values, color=colors[:len(model_names)], edgecolor='black', linewidth=1.5)
        ax.set_title(metric_name, fontweight='bold', fontsize=13)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, max(values) * 1.25)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Hide the last subplot (8th position)
    ax_empty = fig.add_subplot(gs_bars[1, 3])
    ax_empty.axis('off')

    # ========== SECTION 2: DISTRIBUTION HISTOGRAMS FOR EACH MODEL ==========
    metric_keys = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge1', 'rougeL']
    metric_labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-1', 'ROUGE-L']
    hist_colors = ['steelblue', 'steelblue', 'steelblue', 'steelblue', 'green', 'orange', 'orange']

    for model_idx, (model_name, metrics) in enumerate(all_model_metrics.items()):
        # Create subplot grid for this model
        gs_model = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[model_idx + 1],
                                                     wspace=0.3)

        # Add model name as text
        for metric_idx, (metric_key, metric_label, color) in enumerate(zip(metric_keys, metric_labels, hist_colors)):
            ax = fig.add_subplot(gs_model[0, metric_idx])

            # Plot histogram
            ax.hist(metrics[metric_key], bins=25, color=color, alpha=0.7,
                   edgecolor='black', linewidth=1)

            # Add mean line
            mean_val = np.mean(metrics[metric_key])
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
                      label=f'μ={mean_val:.3f}')

            # Formatting
            if metric_idx == 0:
                ax.set_ylabel(f'{model_name.upper()}\nFrequency',
                            fontsize=11, fontweight='bold')

            ax.set_xlabel(metric_label, fontsize=9, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=8)

    # Save the comprehensive report
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved comprehensive report to {output_path}")
    plt.close()

    print(f"✓ Report generated successfully!")


def main():
    """Main evaluation pipeline."""

    # Load data
    print("Loading test data...")
    _, test_data, _, _ = read_data()
    print(f"Test dataset size: {len(test_data)}\n")

    # Define models to evaluate
    models_config = [
        ('genechat', GeneChatModel, f'{MODEL_WEIGHTS_DIR}/genechat_best.pt'),
        ('bart', DNABERTBartDecoder, f'{MODEL_WEIGHTS_DIR}/bart_best.pt'),
        ('t5', DNABERTT5Decoder, f'{MODEL_WEIGHTS_DIR}/t5_best.pt'),
        ('gru', DNABERTGRUDecoder, f'{MODEL_WEIGHTS_DIR}/gru_best.pt'),
    ]

    # Check for random baseline
    random_path = f'{MODEL_WEIGHTS_DIR}/random_final.pt'
    if os.path.exists(random_path):
        models_config.append(('random', RandomSummaryBaseline, random_path))

    # Load all models
    print(f"{'='*80}")
    print("LOADING MODELS")
    print(f"{'='*80}\n")

    models = {}
    for model_name, model_class, checkpoint_path in models_config:
        if os.path.exists(checkpoint_path):
            models[model_name] = load_model(model_name, model_class, checkpoint_path)
        else:
            print(f"⚠ Warning: {checkpoint_path} not found, skipping {model_name}")

    print(f"\n✓ Loaded {len(models)} models: {list(models.keys())}")

    # Evaluate all models
    all_model_metrics = {}
    results_summary = {}

    for model_name, model in models.items():
        metrics = evaluate_model(model, test_data, model_name)
        all_model_metrics[model_name] = metrics

        # Compute averages
        results_summary[model_name] = {
            'BLEU-1': np.mean(metrics['bleu1']),
            'BLEU-2': np.mean(metrics['bleu2']),
            'BLEU-3': np.mean(metrics['bleu3']),
            'BLEU-4': np.mean(metrics['bleu4']),
            'METEOR': np.mean(metrics['meteor']),
            'ROUGE-1': np.mean(metrics['rouge1']),
            'ROUGE-L': np.mean(metrics['rougeL']),
        }

    # Print results table
    print(f"\n{'='*80}")
    print("AVERAGE METRICS COMPARISON")
    print(f"{'='*80}\n")

    df = pd.DataFrame(results_summary).T
    print(df.to_string())
    print()

    # Save results to CSV
    df.to_csv('model_evaluation_results.csv')
    print("✓ Saved results to model_evaluation_results.csv\n")

    # Generate comprehensive report
    generate_comprehensive_report(results_summary, all_model_metrics)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}\n")
    print("Generated files:")
    print("  - evaluation_report.png (Comprehensive report with all visualizations)")
    print("  - model_evaluation_results.csv (Numerical results table)")


if __name__ == "__main__":
    main()
