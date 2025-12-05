#!/usr/bin/env python3
"""
Generate updated evaluation report with:
- Bar graph comparison with genechat renamed to gpt2 and random added
- Distribution plot for random model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# Read the CSV with average results
df = pd.read_csv('model_evaluation_results.csv', index_col=0)
print("Loaded results from CSV:")
print(df)
print()

# Load random model's individual metrics
random_metrics = torch.load('random_metrics.pt')
print("Loaded random model metrics with", len(random_metrics['bleu1']), "samples")

# Prepare data for bar charts
results_summary = {}
for model_name in df.index:
    results_summary[model_name] = {
        'BLEU-1': df.loc[model_name, 'BLEU-1'],
        'BLEU-2': df.loc[model_name, 'BLEU-2'],
        'BLEU-3': df.loc[model_name, 'BLEU-3'],
        'BLEU-4': df.loc[model_name, 'BLEU-4'],
        'METEOR': df.loc[model_name, 'METEOR'],
        'ROUGE-1': df.loc[model_name, 'ROUGE-1'],
        'ROUGE-L': df.loc[model_name, 'ROUGE-L'],
    }

# Generate the report
print("\nGenerating updated evaluation report...")

# Create figure with two sections:
# 1. Bar charts (top)
# 2. Distribution for random model (bottom)
fig = plt.figure(figsize=(24, 14))

gs = gridspec.GridSpec(2, 1, height_ratios=[4, 6], hspace=0.4, figure=fig)

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
    bars = ax.bar(model_names, values, color=colors[:len(model_names)],
                  edgecolor='black', linewidth=1.5)
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

# ========== SECTION 2: DISTRIBUTION HISTOGRAMS FOR RANDOM MODEL ==========
gs_random = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=gs[1], wspace=0.3)

metric_keys = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge1', 'rougeL']
metric_labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-1', 'ROUGE-L']
hist_colors = ['steelblue', 'steelblue', 'steelblue', 'steelblue', 'green', 'orange', 'orange']

for metric_idx, (metric_key, metric_label, color) in enumerate(zip(metric_keys, metric_labels, hist_colors)):
    ax = fig.add_subplot(gs_random[0, metric_idx])

    # Plot histogram
    ax.hist(random_metrics[metric_key], bins=25, color=color, alpha=0.7,
           edgecolor='black', linewidth=1)

    # Add mean line
    mean_val = np.mean(random_metrics[metric_key])
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
              label=f'μ={mean_val:.3f}')

    # Formatting
    if metric_idx == 0:
        ax.set_ylabel('RANDOM\nFrequency', fontsize=11, fontweight='bold')

    ax.set_xlabel(metric_label, fontsize=9, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)

# Save the report
output_path = 'evaluation_report.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved updated report to {output_path}")
plt.close()

print("✓ Report generated successfully!")
