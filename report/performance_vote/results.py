"""
LLM Voting Performance Analysis and Visualization Script
Generates visualizations and summaries for each LLM and dataset
Also analyzes agreement between LLMs on the same images
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Configuration
BASE_DIR = Path(r"C:\Users\binbi\Desktop\DataLab2Project")
PERF_DIR = BASE_DIR / "report" / "performance_vote"
VIZ_DIR = BASE_DIR / "visualizations" / "vote"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["COCO", "Flickr", "VizWiz"]

# Visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_performance_data(dataset_name):
    """Load performance data for a dataset"""
    dataset_dir = PERF_DIR / dataset_name
    if not dataset_dir.exists():
        print(f"⚠️  Directory {dataset_dir} not found")
        return None
    
    # Find the most recent performance file
    perf_files = list(dataset_dir.glob("llm_voting_performance_*.json"))
    if not perf_files:
        print(f"⚠️  No performance file found for {dataset_name}")
        return None
    
    latest_file = max(perf_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Loading {latest_file.name} for {dataset_name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_llm_performance_comparison(all_data):
    """Create performance comparison of all LLMs across all datasets"""
    
    # Collect data
    performance_data = []
    for dataset_name, data in all_data.items():
        if data is None:
            continue
        
        for llm_name, llm_eval in data['llm_evaluations'].items():
            if llm_eval['status'] != 'completed':
                continue
            
            no_degradation = llm_eval['improvements'] + llm_eval['no_changes']
            no_degradation_rate = (no_degradation / llm_eval['successful_evaluations']) * 100 if llm_eval['successful_evaluations'] > 0 else 0
            
            performance_data.append({
                'Dataset': dataset_name,
                'LLM': llm_name,
                'Improvement Rate': llm_eval['improvement_rate'] * 100,
                'Degradation Rate': llm_eval['degradation_rate'] * 100,
                'Stability Rate': (llm_eval['no_changes'] / llm_eval['successful_evaluations']) * 100 if llm_eval['successful_evaluations'] > 0 else 0,
                'No Degradation Rate': no_degradation_rate,
                'Successful Evaluations': llm_eval['successful_evaluations'],
                'Improvements': llm_eval['improvements'],
                'Degradations': llm_eval['degradations'],
                'No Change': llm_eval['no_changes'],
                'No Degradation': no_degradation
            })
    
    if not performance_data:
        print("⚠️  No performance data to visualize")
        return
    
    df = pd.DataFrame(performance_data)
    
    # 1. Grouped bar chart - Improvement rate by LLM and dataset
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df['Dataset'].unique()))
    width = 0.2
    llms = df['LLM'].unique()
    
    for i, llm in enumerate(llms):
        llm_data = df[df['LLM'] == llm]
        offset = width * (i - len(llms)/2 + 0.5)
        ax.bar(x + offset, llm_data['Improvement Rate'], width, 
               label=llm, alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Improvement Rates by LLM and Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Dataset'].unique())
    ax.legend(title='LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'comparison_improvement_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1b. Bar chart - No Degradation Rate comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, llm in enumerate(llms):
        llm_data = df[df['LLM'] == llm]
        offset = width * (i - len(llms)/2 + 0.5)
        ax.bar(x + offset, llm_data['No Degradation Rate'], width, 
               label=llm, alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('No Degradation Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of No Degradation Rates by LLM and Dataset\n(Same rank or better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Dataset'].unique())
    ax.legend(title='LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'comparison_no_degradation_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics_to_plot = ['Improvement Rate', 'Degradation Rate', 'Stability Rate', 'No Degradation Rate']
    
    for idx, metric in enumerate(metrics_to_plot):
        pivot_df = df.pivot(index='LLM', columns='Dataset', values=metric)
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[idx], cbar_kws={'label': '%'}, vmin=0, vmax=100)
        axes[idx].set_title(metric, fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
    
    plt.suptitle('Performance Metrics Heatmap by LLM and Dataset', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'heatmap_performances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Stacked bar chart - Results distribution
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_sorted = df.sort_values(['Dataset', 'LLM'])
    datasets_llms = [f"{row['Dataset']}\n{row['LLM'].split(':')[0]}" 
                     for _, row in df_sorted.iterrows()]
    
    width = 0.6
    x_pos = np.arange(len(datasets_llms))
    
    p1 = ax.bar(x_pos, df_sorted['Improvement Rate'], width, 
                label='Improvement', color='#2ecc71', alpha=0.8)
    p2 = ax.bar(x_pos, df_sorted['Stability Rate'], width,
                bottom=df_sorted['Improvement Rate'],
                label='Stable', color='#95a5a6', alpha=0.8)
    p3 = ax.bar(x_pos, df_sorted['Degradation Rate'], width,
                bottom=df_sorted['Improvement Rate'] + df_sorted['Stability Rate'],
                label='Degradation', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Results Distribution by LLM and Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets_llms, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'stacked_results_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save summary table
    df.to_csv(VIZ_DIR / 'performance_summary.csv', index=False)
    
    # 5. Global performance across all LLMs and datasets
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 5a. Average performance by LLM (across all datasets)
    llm_avg = df.groupby('LLM')[['Improvement Rate', 'Degradation Rate', 'Stability Rate', 'No Degradation Rate']].mean()
    
    ax = axes[0, 0]
    llm_avg.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Average Performance by LLM (Across All Datasets)', fontweight='bold', fontsize=12)
    ax.set_xlabel('LLM', fontweight='bold')
    ax.legend(title='Metric', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 5b. Average performance by Dataset (across all LLMs)
    dataset_avg = df.groupby('Dataset')[['Improvement Rate', 'Degradation Rate', 'Stability Rate', 'No Degradation Rate']].mean()
    
    ax = axes[0, 1]
    dataset_avg.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Average Performance by Dataset (Across All LLMs)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.legend(title='Metric', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 5c. Overall global performance
    global_avg = df[['Improvement Rate', 'Degradation Rate', 'Stability Rate', 'No Degradation Rate']].mean()
    
    ax = axes[1, 0]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6', '#3498db']
    bars = ax.bar(range(len(global_avg)), global_avg.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Global Performance (All LLMs & Datasets)', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(global_avg)))
    ax.set_xticklabels(['Improvement', 'Degradation', 'Stability', 'No Degradation'], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(global_avg.values) * 1.15)
    
    # Add values on bars
    for bar, val in zip(bars, global_avg.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 5d. Total evaluations statistics
    ax = axes[1, 1]
    total_evals = df.groupby('LLM')['Successful Evaluations'].sum().sort_values(ascending=True)
    total_evals.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
    ax.set_xlabel('Total Successful Evaluations', fontweight='bold')
    ax.set_title('Total Evaluations by LLM', fontweight='bold', fontsize=12)
    ax.set_ylabel('LLM', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (llm, val) in enumerate(total_evals.items()):
        ax.text(val + max(total_evals) * 0.01, i, f'{int(val):,}', 
               va='center', fontweight='bold', fontsize=9)
    
    plt.suptitle('Global Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'global_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Rankings visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 6a. Ranking by Improvement Rate
    ax = axes[0]
    improvement_ranking = df.groupby('LLM')['Improvement Rate'].mean().sort_values(ascending=True)
    bars = ax.barh(range(len(improvement_ranking)), improvement_ranking.values, 
                   color=plt.cm.RdYlGn(improvement_ranking.values / 100), alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(improvement_ranking)))
    ax.set_yticklabels(improvement_ranking.index)
    ax.set_xlabel('Average Improvement Rate (%)', fontweight='bold')
    ax.set_title('LLM Ranking by Improvement Rate', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Add values and ranks
    for i, (llm, val) in enumerate(improvement_ranking.items()):
        rank = len(improvement_ranking) - i
        ax.text(val + max(improvement_ranking) * 0.01, i, 
               f'#{rank}: {val:.2f}%', va='center', fontweight='bold', fontsize=9)
    
    # 6b. Ranking by No Degradation Rate
    ax = axes[1]
    no_degrade_ranking = df.groupby('LLM')['No Degradation Rate'].mean().sort_values(ascending=True)
    bars = ax.barh(range(len(no_degrade_ranking)), no_degrade_ranking.values, 
                   color=plt.cm.RdYlGn(no_degrade_ranking.values / 100), alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(no_degrade_ranking)))
    ax.set_yticklabels(no_degrade_ranking.index)
    ax.set_xlabel('Average No Degradation Rate (%)', fontweight='bold')
    ax.set_title('LLM Ranking by No Degradation Rate\n(Same rank or better)', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Add values and ranks
    for i, (llm, val) in enumerate(no_degrade_ranking.items()):
        rank = len(no_degrade_ranking) - i
        ax.text(val + max(no_degrade_ranking) * 0.01, i, 
               f'#{rank}: {val:.2f}%', va='center', fontweight='bold', fontsize=9)
    
    plt.suptitle('LLM Rankings', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'llm_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Multi-bar stacked visualization by Dataset and LLM
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Prepare data grouped by dataset
    datasets = df['Dataset'].unique()
    llms = df['LLM'].unique()
    
    # Sort LLMs by model size (extract number from name like "7b", "20b", etc.)
    def extract_model_size(llm_name):
        import re
        match = re.search(r'(\d+)b', llm_name.lower())
        if match:
            return int(match.group(1))
        return 0  # Default for models without size
    
    llms_sorted = sorted(llms, key=extract_model_size)
    
    x_labels = []
    
    # Create position for each bar
    bar_width = 0.8
    group_width = len(llms) * bar_width
    group_spacing = 0.5
    
    x_positions = []
    current_x = 0
    
    improvement_data = []
    no_change_data = []
    degradation_data = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        for llm in llms_sorted:  # Use sorted LLMs
            llm_data = dataset_df[dataset_df['LLM'] == llm]
            if not llm_data.empty:
                improvement_data.append(llm_data['Improvement Rate'].values[0])
                no_change_data.append(llm_data['Stability Rate'].values[0])
                degradation_data.append(llm_data['Degradation Rate'].values[0])
                x_positions.append(current_x)
                x_labels.append(f"{llm}")  # Full LLM name
                current_x += bar_width
        current_x += group_spacing  # Space between dataset groups
    
    # Create stacked bars
    p1 = ax.bar(x_positions, degradation_data, bar_width, 
                label='Degradation', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
    p2 = ax.bar(x_positions, no_change_data, bar_width, 
                bottom=degradation_data,
                label='No Change', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Calculate bottom for improvement (degradation + no_change)
    bottom_improvement = [d + n for d, n in zip(degradation_data, no_change_data)]
    p3 = ax.bar(x_positions, improvement_data, bar_width,
                bottom=bottom_improvement,
                label='Improvement', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add dataset labels and separators
    current_x = 0
    for i, dataset in enumerate(datasets):
        dataset_llms_count = len(llms_sorted)  # Use sorted LLMs count
        dataset_center = current_x + (dataset_llms_count * bar_width) / 2 - bar_width / 2
        
        # Add dataset label below x-axis
        ax.text(dataset_center, -8, dataset, 
               ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        current_x += dataset_llms_count * bar_width + group_spacing
    
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Distribution by Dataset and LLM\n(Stacked: Degradation + No Change + Improvement)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(-10, 105)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.5, max(x_positions) + 1)
    
    # Add horizontal dashed line at 50% (in foreground, above bars)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=2.5, alpha=0.9, zorder=10)
    ax.text(max(x_positions) + 0.5, 50, '50%', va='center', ha='left', 
           fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9), zorder=10)
    
    # Add percentage labels on bars (only for segments > 5%)
    for i, (x_pos, imp, nc, deg) in enumerate(zip(x_positions, improvement_data, no_change_data, degradation_data)):
        # Degradation label
        if deg > 5:
            ax.text(x_pos, deg/2, f'{deg:.0f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        # No change label
        if nc > 5:
            ax.text(x_pos, deg + nc/2, f'{nc:.0f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        # Improvement label
        if imp > 5:
            ax.text(x_pos, deg + nc + imp/2, f'{imp:.0f}%', 
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'multi_bar_stacked_by_dataset_llm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison visualizations saved in {VIZ_DIR}")
    
    # Print global statistics
    print(f"\n{'='*60}")
    print("GLOBAL STATISTICS (All LLMs & Datasets)")
    print(f"{'='*60}")
    print(f"Total evaluations: {df['Successful Evaluations'].sum():,}")
    print(f"Total improvements: {df['Improvements'].sum():,}")
    print(f"Total degradations: {df['Degradations'].sum():,}")
    print(f"Total no change: {df['No Change'].sum():,}")
    print(f"Total no degradation: {df['No Degradation'].sum():,}")
    print(f"\nAverage improvement rate: {global_avg['Improvement Rate']:.2f}%")
    print(f"Average degradation rate: {global_avg['Degradation Rate']:.2f}%")
    print(f"Average stability rate: {global_avg['Stability Rate']:.2f}%")
    print(f"Average no degradation rate: {global_avg['No Degradation Rate']:.2f}%")
    print(f"{'='*60}\n")
    
    return df


def analyze_performance_by_rank(all_data):
    """Analyze performance by expected rank across all LLMs and datasets"""
    
    print("\n📊 Analyzing performance by rank...")
    
    # Collect data by rank
    rank_data = defaultdict(lambda: {
        'improvements': 0,
        'degradations': 0,
        'no_change': 0,
        'total': 0,
        'llm_details': defaultdict(lambda: {'improvements': 0, 'degradations': 0, 'no_change': 0, 'total': 0}),
        'dataset_details': defaultdict(lambda: {'improvements': 0, 'degradations': 0, 'no_change': 0, 'total': 0})
    })
    
    for dataset_name, data in all_data.items():
        if data is None:
            continue
        
        for llm_name, llm_eval in data['llm_evaluations'].items():
            if llm_eval['status'] != 'completed':
                continue
            
            for result in llm_eval.get('results', []):
                rank = result.get('expected_rank', 0)
                improvement = result.get('improvement', 0)
                
                rank_data[rank]['total'] += 1
                rank_data[rank]['llm_details'][llm_name]['total'] += 1
                rank_data[rank]['dataset_details'][dataset_name]['total'] += 1
                
                if improvement > 0:
                    rank_data[rank]['improvements'] += 1
                    rank_data[rank]['llm_details'][llm_name]['improvements'] += 1
                    rank_data[rank]['dataset_details'][dataset_name]['improvements'] += 1
                elif improvement < 0:
                    rank_data[rank]['degradations'] += 1
                    rank_data[rank]['llm_details'][llm_name]['degradations'] += 1
                    rank_data[rank]['dataset_details'][dataset_name]['degradations'] += 1
                else:
                    rank_data[rank]['no_change'] += 1
                    rank_data[rank]['llm_details'][llm_name]['no_change'] += 1
                    rank_data[rank]['dataset_details'][dataset_name]['no_change'] += 1
    
    if not rank_data:
        print("⚠️  No rank data available")
        return
    
    ranks = sorted(rank_data.keys())
    
    # 1. Overall performance by rank
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1a. Stacked bar chart by rank
    ax = axes[0, 0]
    improvements_pct = [rank_data[r]['improvements'] / rank_data[r]['total'] * 100 for r in ranks]
    no_change_pct = [rank_data[r]['no_change'] / rank_data[r]['total'] * 100 for r in ranks]
    degradations_pct = [rank_data[r]['degradations'] / rank_data[r]['total'] * 100 for r in ranks]
    
    x = np.arange(len(ranks))
    width = 0.6
    
    p1 = ax.bar(x, degradations_pct, width, label='Degradation', color='#e74c3c', alpha=0.85)
    p2 = ax.bar(x, no_change_pct, width, bottom=degradations_pct,
                label='No Change', color='#f39c12', alpha=0.85)
    bottom_imp = [d + n for d, n in zip(degradations_pct, no_change_pct)]
    p3 = ax.bar(x, improvements_pct, width, bottom=bottom_imp,
                label='Improvement', color='#2ecc71', alpha=0.85)
    
    ax.set_xlabel('Expected Rank', fontweight='bold', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax.set_title('Overall Performance by Expected Rank', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rank {r}' for r in ranks])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # 1b. Line chart - Improvement rate trend
    ax = axes[0, 1]
    ax.plot(ranks, improvements_pct, marker='o', linewidth=2.5, markersize=10, 
            color='#2ecc71', label='Improvement Rate')
    ax.plot(ranks, degradations_pct, marker='s', linewidth=2.5, markersize=10, 
            color='#e74c3c', label='Degradation Rate')
    no_degrade_pct = [i + n for i, n in zip(improvements_pct, no_change_pct)]
    ax.plot(ranks, no_degrade_pct, marker='^', linewidth=2.5, markersize=10, 
            color='#3498db', label='No Degradation Rate')
    
    ax.set_xlabel('Expected Rank', fontweight='bold', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax.set_title('Performance Trends by Expected Rank', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)
    ax.set_xticklabels([f'Rank {r}' for r in ranks])
    
    # 1c. Count of evaluations by rank
    ax = axes[1, 0]
    counts = [rank_data[r]['total'] for r in ranks]
    bars = ax.bar(ranks, counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Expected Rank', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Evaluations', fontweight='bold', fontsize=12)
    ax.set_title('Total Evaluations by Expected Rank', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(ranks)
    ax.set_xticklabels([f'Rank {r}' for r in ranks])
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 1d. Grouped bar chart - Absolute numbers
    ax = axes[1, 1]
    improvements_count = [rank_data[r]['improvements'] for r in ranks]
    no_change_count = [rank_data[r]['no_change'] for r in ranks]
    degradations_count = [rank_data[r]['degradations'] for r in ranks]
    
    x = np.arange(len(ranks))
    width = 0.25
    
    ax.bar(x - width, improvements_count, width, label='Improvements', color='#2ecc71', alpha=0.8)
    ax.bar(x, no_change_count, width, label='No Change', color='#f39c12', alpha=0.8)
    ax.bar(x + width, degradations_count, width, label='Degradations', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Expected Rank', fontweight='bold', fontsize=12)
    ax.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax.set_title('Absolute Counts by Expected Rank', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rank {r}' for r in ranks])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Performance Analysis by Expected Rank (All LLMs & Datasets)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'performance_by_rank_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance by rank for each LLM
    llm_names = set()
    for rank in ranks:
        llm_names.update(rank_data[rank]['llm_details'].keys())
    llm_names = sorted(llm_names)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    # Sort LLMs by size
    def extract_model_size(llm_name):
        import re
        match = re.search(r'(\d+)b', llm_name.lower())
        if match:
            return int(match.group(1))
        return 0
    
    llm_names_sorted = sorted(llm_names, key=extract_model_size)
    
    for idx, llm in enumerate(llm_names_sorted):
        ax = axes[idx]
        
        llm_improvements = []
        llm_no_change = []
        llm_degradations = []
        
        for rank in ranks:
            details = rank_data[rank]['llm_details'][llm]
            total = details['total']
            if total > 0:
                llm_improvements.append(details['improvements'] / total * 100)
                llm_no_change.append(details['no_change'] / total * 100)
                llm_degradations.append(details['degradations'] / total * 100)
            else:
                llm_improvements.append(0)
                llm_no_change.append(0)
                llm_degradations.append(0)
        
        x = np.arange(len(ranks))
        width = 0.6
        
        p1 = ax.bar(x, llm_degradations, width, label='Degradation', color='#e74c3c', alpha=0.85)
        p2 = ax.bar(x, llm_no_change, width, bottom=llm_degradations,
                    label='No Change', color='#f39c12', alpha=0.85)
        bottom = [d + n for d, n in zip(llm_degradations, llm_no_change)]
        p3 = ax.bar(x, llm_improvements, width, bottom=bottom,
                    label='Improvement', color='#2ecc71', alpha=0.85)
        
        ax.set_xlabel('Expected Rank', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title(f'{llm}', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'R{r}' for r in ranks])
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.suptitle('Performance by Expected Rank - Individual LLM Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'performance_by_rank_per_llm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance by rank for each dataset
    dataset_names = set()
    for rank in ranks:
        dataset_names.update(rank_data[rank]['dataset_details'].keys())
    dataset_names = sorted(dataset_names)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, dataset in enumerate(dataset_names):
        ax = axes[idx]
        
        dataset_improvements = []
        dataset_no_change = []
        dataset_degradations = []
        
        for rank in ranks:
            details = rank_data[rank]['dataset_details'][dataset]
            total = details['total']
            if total > 0:
                dataset_improvements.append(details['improvements'] / total * 100)
                dataset_no_change.append(details['no_change'] / total * 100)
                dataset_degradations.append(details['degradations'] / total * 100)
            else:
                dataset_improvements.append(0)
                dataset_no_change.append(0)
                dataset_degradations.append(0)
        
        x = np.arange(len(ranks))
        width = 0.6
        
        p1 = ax.bar(x, dataset_degradations, width, label='Degradation', color='#e74c3c', alpha=0.85)
        p2 = ax.bar(x, dataset_no_change, width, bottom=dataset_degradations,
                    label='No Change', color='#f39c12', alpha=0.85)
        bottom = [d + n for d, n in zip(dataset_degradations, dataset_no_change)]
        p3 = ax.bar(x, dataset_improvements, width, bottom=bottom,
                    label='Improvement', color='#2ecc71', alpha=0.85)
        
        ax.set_xlabel('Expected Rank', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title(f'{dataset}', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Rank {r}' for r in ranks])
        if idx == 0:
            ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.suptitle('Performance by Expected Rank - Dataset Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'performance_by_rank_per_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap: LLM vs Rank (Improvement Rate)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_data = []
    for llm in llm_names_sorted:
        llm_row = []
        for rank in ranks:
            details = rank_data[rank]['llm_details'][llm]
            total = details['total']
            if total > 0:
                imp_rate = details['improvements'] / total * 100
            else:
                imp_rate = 0
            llm_row.append(imp_rate)
        heatmap_data.append(llm_row)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
               xticklabels=[f'Rank {r}' for r in ranks],
               yticklabels=llm_names_sorted,
               ax=ax, cbar_kws={'label': 'Improvement Rate (%)'}, vmin=0, vmax=100)
    
    ax.set_title('Improvement Rate Heatmap: LLM vs Expected Rank', 
                fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Expected Rank', fontweight='bold')
    ax.set_ylabel('LLM', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'heatmap_llm_vs_rank.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Rank analysis visualizations saved")
    
    # Print rank statistics
    print(f"\n{'='*60}")
    print("PERFORMANCE BY RANK STATISTICS")
    print(f"{'='*60}")
    for rank in ranks:
        total = rank_data[rank]['total']
        imp = rank_data[rank]['improvements']
        deg = rank_data[rank]['degradations']
        nc = rank_data[rank]['no_change']
        print(f"\nRank {rank}:")
        print(f"  Total evaluations: {total:,}")
        print(f"  Improvements: {imp:,} ({imp/total*100:.2f}%)")
        print(f"  Degradations: {deg:,} ({deg/total*100:.2f}%)")
        print(f"  No change: {nc:,} ({nc/total*100:.2f}%)")
        print(f"  No degradation: {imp+nc:,} ({(imp+nc)/total*100:.2f}%)")
    print(f"{'='*60}\n")



def create_llm_specific_visualizations(dataset_name, data):
    """Create detailed visualizations for each LLM of a dataset"""
    
    if data is None:
        return
    
    dataset_viz_dir = VIZ_DIR / dataset_name
    dataset_viz_dir.mkdir(parents=True, exist_ok=True)
    
    for llm_name, llm_eval in data['llm_evaluations'].items():
        if llm_eval['status'] != 'completed':
            continue
        
        llm_safe_name = llm_name.replace(':', '_').replace('/', '_')
        
        # 1. Pie chart of results distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sizes = [llm_eval['improvements'], llm_eval['degradations'], llm_eval['no_changes']]
        labels = ['Improvements', 'Degradations', 'No Change']
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        explode = (0.1, 0, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.set_title(f'Results Distribution - {llm_name}\n{dataset_name}', 
                     fontweight='bold')
        
        # 2. Bar chart of metrics
        no_degradation = llm_eval['improvements'] + llm_eval['no_changes']
        no_degradation_rate = (no_degradation / llm_eval['successful_evaluations']) * 100
        
        metrics = {
            'Improvement\nRate': llm_eval['improvement_rate'] * 100,
            'Degradation\nRate': llm_eval['degradation_rate'] * 100,
            'Stability\nRate': (llm_eval['no_changes'] / llm_eval['successful_evaluations']) * 100,
            'No Degradation\nRate': no_degradation_rate
        }
        
        bars = ax2.bar(metrics.keys(), metrics.values(), 
                      color=['#2ecc71', '#e74c3c', '#95a5a6', '#3498db'], alpha=0.7)
        ax2.set_ylabel('Percentage (%)', fontweight='bold')
        ax2.set_title(f'Performance Metrics - {llm_name}', fontweight='bold')
        ax2.set_ylim(0, max(metrics.values()) * 1.2)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(dataset_viz_dir / f'{llm_safe_name}_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Analysis by expected rank
        rank_analysis = defaultdict(lambda: {'improvements': 0, 'degradations': 0, 'no_change': 0, 'total': 0})
        
        for result in llm_eval.get('results', []):
            rank = result.get('expected_rank', 0)
            improvement = result.get('improvement', 0)
            
            rank_analysis[rank]['total'] += 1
            if improvement > 0:
                rank_analysis[rank]['improvements'] += 1
            elif improvement < 0:
                rank_analysis[rank]['degradations'] += 1
            else:
                rank_analysis[rank]['no_change'] += 1
        
        if rank_analysis:
            ranks = sorted(rank_analysis.keys())
            improvements_pct = [rank_analysis[r]['improvements'] / rank_analysis[r]['total'] * 100 for r in ranks]
            degradations_pct = [rank_analysis[r]['degradations'] / rank_analysis[r]['total'] * 100 for r in ranks]
            stable_pct = [rank_analysis[r]['no_change'] / rank_analysis[r]['total'] * 100 for r in ranks]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(ranks))
            width = 0.25
            
            ax.bar(x - width, improvements_pct, width, label='Improvements', color='#2ecc71', alpha=0.8)
            ax.bar(x, stable_pct, width, label='Stable', color='#95a5a6', alpha=0.8)
            ax.bar(x + width, degradations_pct, width, label='Degradations', color='#e74c3c', alpha=0.8)
            
            ax.set_xlabel('Expected Rank', fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontweight='bold')
            ax.set_title(f'Performance by Expected Rank - {llm_name} ({dataset_name})', 
                        fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(ranks)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(dataset_viz_dir / f'{llm_safe_name}_by_rank.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✅ Visualizations created for {llm_name}")


def analyze_llm_agreement(all_data):
    """Analyze agreement between LLMs on the same images"""
    
    print("\n🔍 Analyzing agreement between LLMs...")
    
    for dataset_name, data in all_data.items():
        if data is None:
            continue
        
        print(f"\n📊 Dataset: {dataset_name}")
        
        # Collect each LLM's decisions for each image
        image_decisions = defaultdict(dict)
        llm_names = []
        
        for llm_name, llm_eval in data['llm_evaluations'].items():
            if llm_eval['status'] != 'completed':
                continue
            
            llm_names.append(llm_name)
            
            for result in llm_eval.get('results', []):
                image_id = result['image_id']
                improvement = result.get('improvement', 0)
                
                # Categorize decision
                if improvement > 0:
                    decision = 'improvement'
                elif improvement < 0:
                    decision = 'degradation'
                else:
                    decision = 'stable'
                
                image_decisions[image_id][llm_name] = {
                    'decision': decision,
                    'improvement': improvement,
                    'clip_rank': result.get('clip_rank'),
                    'llm_rank': result.get('llm_rank')
                }
        
        if len(llm_names) < 2:
            print("  ⚠️  Not enough LLMs to analyze agreement")
            continue
        
        # Calculate agreements
        agreement_matrix = np.zeros((len(llm_names), len(llm_names)))
        pairwise_agreements = defaultdict(list)
        
        for image_id, decisions in image_decisions.items():
            if len(decisions) < 2:
                continue
            
            for i, llm1 in enumerate(llm_names):
                for j, llm2 in enumerate(llm_names):
                    if llm1 in decisions and llm2 in decisions:
                        if decisions[llm1]['decision'] == decisions[llm2]['decision']:
                            agreement_matrix[i, j] += 1
                            if i < j:  # Avoid duplicates
                                pairwise_agreements[f"{llm1} vs {llm2}"].append(1)
                        else:
                            if i < j:
                                pairwise_agreements[f"{llm1} vs {llm2}"].append(0)
        
        # Normalize agreement matrix
        num_images = len(image_decisions)
        if num_images > 0:
            agreement_matrix = (agreement_matrix / num_images) * 100
        
        # Visualization 1: Agreement heatmap between LLMs
        fig, ax = plt.subplots(figsize=(10, 8))
        
        llm_labels = [llm.split(':')[0] for llm in llm_names]
        sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                   xticklabels=llm_labels, yticklabels=llm_labels,
                   ax=ax, cbar_kws={'label': 'Agreement (%)'}, vmin=0, vmax=100)
        
        ax.set_title(f'Agreement Matrix Between LLMs - {dataset_name}\n(% of images with same decision)', 
                    fontweight='bold', pad=20, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(VIZ_DIR / f'{dataset_name}_llm_agreement_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualization 2: Pairwise agreement
        if pairwise_agreements:
            pair_names = list(pairwise_agreements.keys())
            pair_scores = [np.mean(agreements) * 100 for agreements in pairwise_agreements.values()]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(pair_names, pair_scores, color='steelblue', alpha=0.7)
            
            ax.set_xlabel('Agreement Rate (%)', fontweight='bold')
            ax.set_title(f'Pairwise LLM Agreement - {dataset_name}', 
                        fontweight='bold', fontsize=14)
            ax.set_xlim(0, 100)
            ax.grid(axis='x', alpha=0.3)
            
            # Add values
            for bar, score in zip(bars, pair_scores):
                ax.text(score + 1, bar.get_y() + bar.get_height()/2, 
                       f'{score:.1f}%', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(VIZ_DIR / f'{dataset_name}_pairwise_agreement.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Analysis of consensus and disagreement cases
        full_agreement = 0
        full_disagreement = 0
        partial_agreement = 0
        
        for image_id, decisions in image_decisions.items():
            if len(decisions) < len(llm_names):
                continue
            
            decision_types = [d['decision'] for d in decisions.values()]
            unique_decisions = set(decision_types)
            
            if len(unique_decisions) == 1:
                full_agreement += 1
            elif len(unique_decisions) == len(decision_types):
                full_disagreement += 1
            else:
                partial_agreement += 1
        
        total_images = full_agreement + full_disagreement + partial_agreement
        
        if total_images > 0:
            # Visualization 3: Global consensus
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = ['Full\nConsensus', 'Partial\nAgreement', 'Full\nDisagreement']
            values = [full_agreement, partial_agreement, full_disagreement]
            percentages = [v / total_images * 100 for v in values]
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            
            bars = ax.bar(categories, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.set_ylabel('Percentage of Images (%)', fontweight='bold')
            ax.set_title(f'Agreement Level Between All LLMs - {dataset_name}\n({total_images} images analyzed)', 
                        fontweight='bold', fontsize=14, pad=20)
            ax.set_ylim(0, max(percentages) * 1.2)
            
            # Add values
            for bar, pct, val in zip(bars, percentages, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%\n({val} images)', 
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(VIZ_DIR / f'{dataset_name}_global_consensus.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Full consensus: {full_agreement} images ({full_agreement/total_images*100:.1f}%)")
            print(f"  ⚠️  Partial agreement: {partial_agreement} images ({partial_agreement/total_images*100:.1f}%)")
            print(f"  ❌ Full disagreement: {full_disagreement} images ({full_disagreement/total_images*100:.1f}%)")


def generate_summary_report(all_data, performance_df):
    """Generate a text summary report"""
    
    report_path = VIZ_DIR / 'performance_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM VOTING PERFORMANCE REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_name, data in all_data.items():
            if data is None:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"DATASET: {dataset_name}\n")
            f.write(f"{'='*80}\n\n")
            
            metadata = data.get('metadata', {})
            f.write(f"Analysis period: {metadata.get('start_time', 'N/A')} → {metadata.get('completion_time', 'N/A')}\n")
            f.write(f"Total test cases: {metadata.get('total_selected_cases', 'N/A')}\n")
            f.write(f"Target ranks: {metadata.get('target_ranks', 'N/A')}\n")
            f.write(f"Total execution time: {metadata.get('total_elapsed_time_seconds', 0) / 3600:.2f} hours\n\n")
            
            for llm_name, llm_eval in data['llm_evaluations'].items():
                if llm_eval['status'] != 'completed':
                    f.write(f"\n⚠️  {llm_name}: {llm_eval['status']}\n")
                    continue
                
                f.write(f"\n{'-'*80}\n")
                f.write(f"LLM: {llm_name}\n")
                f.write(f"{'-'*80}\n\n")
                
                f.write(f"Successful evaluations: {llm_eval['successful_evaluations']:,} / {llm_eval['total_test_cases']:,}\n")
                f.write(f"Success rate: {llm_eval['successful_evaluations'] / llm_eval['total_test_cases'] * 100:.2f}%\n\n")
                
                no_degradation = llm_eval['improvements'] + llm_eval['no_changes']
                no_degradation_rate = (no_degradation / llm_eval['successful_evaluations']) * 100
                
                f.write("Results:\n")
                f.write(f"  • Improvements: {llm_eval['improvements']:,} ({llm_eval['improvement_rate']*100:.2f}%)\n")
                f.write(f"  • Degradations: {llm_eval['degradations']:,} ({llm_eval['degradation_rate']*100:.2f}%)\n")
                f.write(f"  • No change: {llm_eval['no_changes']:,} ({llm_eval['no_changes']/llm_eval['successful_evaluations']*100:.2f}%)\n")
                f.write(f"  • No degradation (same rank or better): {no_degradation:,} ({no_degradation_rate:.2f}%)\n\n")
                
                f.write(f"Execution time: {llm_eval['elapsed_time_seconds']:.2f} seconds\n")
                f.write(f"Average time per evaluation: {llm_eval['elapsed_time_seconds']/llm_eval['successful_evaluations']:.3f} seconds\n")
        
        # Add global rankings
        if performance_df is not None and not performance_df.empty:
            f.write(f"\n{'='*80}\n")
            f.write("GLOBAL STATISTICS (All LLMs & Datasets)\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Total evaluations: {performance_df['Successful Evaluations'].sum():,}\n")
            f.write(f"Total improvements: {performance_df['Improvements'].sum():,}\n")
            f.write(f"Total degradations: {performance_df['Degradations'].sum():,}\n")
            f.write(f"Total no change: {performance_df['No Change'].sum():,}\n")
            f.write(f"Total no degradation: {performance_df['No Degradation'].sum():,}\n\n")
            
            global_avg = performance_df[['Improvement Rate', 'Degradation Rate', 'Stability Rate', 'No Degradation Rate']].mean()
            f.write(f"Average improvement rate: {global_avg['Improvement Rate']:.2f}%\n")
            f.write(f"Average degradation rate: {global_avg['Degradation Rate']:.2f}%\n")
            f.write(f"Average stability rate: {global_avg['Stability Rate']:.2f}%\n")
            f.write(f"Average no degradation rate: {global_avg['No Degradation Rate']:.2f}%\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("GLOBAL RANKING BY IMPROVEMENT RATE\n")
            f.write(f"{'='*80}\n\n")
            
            ranking = performance_df.groupby('LLM')['Improvement Rate'].mean().sort_values(ascending=False)
            for rank, (llm, score) in enumerate(ranking.items(), 1):
                f.write(f"{rank}. {llm}: {score:.2f}%\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("GLOBAL RANKING BY NO DEGRADATION RATE\n")
            f.write(f"{'='*80}\n\n")
            
            ranking_no_degrade = performance_df.groupby('LLM')['No Degradation Rate'].mean().sort_values(ascending=False)
            for rank, (llm, score) in enumerate(ranking_no_degrade.items(), 1):
                f.write(f"{rank}. {llm}: {score:.2f}%\n")
    
    print(f"\n✅ Text report saved: {report_path}")


def main():
    """Main function"""
    print("🚀 Starting LLM voting performance analysis\n")
    print(f"📂 Output directory: {VIZ_DIR}\n")
    
    # Load all data
    all_data = {}
    for dataset in DATASETS:
        print(f"\n📊 Processing dataset: {dataset}")
        data = load_performance_data(dataset)
        all_data[dataset] = data
        
        if data:
            # Create LLM-specific visualizations
            create_llm_specific_visualizations(dataset, data)
    
    # Create global comparative visualizations
    print("\n\n📈 Creating comparative visualizations...")
    performance_df = create_llm_performance_comparison(all_data)
    
    # Analyze agreement between LLMs
    analyze_llm_agreement(all_data)
    
    # Analyze performance by rank
    analyze_performance_by_rank(all_data)
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    generate_summary_report(all_data, performance_df)
    
    print(f"\n\n✨ Analysis complete! All files are in: {VIZ_DIR}")
    print("\nGenerated files:")
    print("  • comparison_improvement_rates.png - Comparison of improvement rates")
    print("  • comparison_no_degradation_rates.png - Comparison of no degradation rates")
    print("  • heatmap_performances.png - Performance metrics heatmap (4 metrics)")
    print("  • stacked_results_distribution.png - Stacked results distribution")
    print("  • multi_bar_stacked_by_dataset_llm.png - Multi-bar stacked by dataset and LLM")
    print("  • global_performance_analysis.png - Global performance across all LLMs and datasets")
    print("  • llm_rankings.png - LLM rankings by improvement and no degradation rates")
    print("  • performance_by_rank_overall.png - Overall performance analysis by rank")
    print("  • performance_by_rank_per_llm.png - Performance by rank for each LLM")
    print("  • performance_by_rank_per_dataset.png - Performance by rank for each dataset")
    print("  • heatmap_llm_vs_rank.png - Improvement rate heatmap: LLM vs Rank")
    print("  • performance_summary.csv - Summary table")
    print("  • {dataset}_llm_agreement_matrix.png - Agreement matrices between LLMs")
    print("  • {dataset}_pairwise_agreement.png - Pairwise LLM agreement")
    print("  • {dataset}_global_consensus.png - Global consensus level")
    print("  • {dataset}/{llm}_overview.png - Overview by LLM (with 4 metrics)")
    print("  • {dataset}/{llm}_by_rank.png - Performance by expected rank")
    print("  • performance_report.txt - Detailed text report with global statistics")


if __name__ == "__main__":
    main()
