"""
Visualization script for retrieval performance across all datasets
Analyzes performance.json and statistics.json files from COCO, Flickr and VizWiz
Generates comparative visualizations in visualizations/raw/
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# File paths
BASE_PATH = Path(r"c:\Users\binbi\Desktop\DataLab2Project")
DATASETS = ["COCO", "Flickr", "VizWiz"]
OUTPUT_DIR = BASE_PATH / "visualizations" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all performance and statistics files"""
    data = {}
    
    for dataset in DATASETS:
        dataset_path = BASE_PATH / "report" / "performance_raw" / dataset
        
        # Load statistics
        stats_file = dataset_path / "statistics.json"
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # Load performance
        perf_file = dataset_path / "performance.json"
        with open(perf_file, 'r', encoding='utf-8') as f:
            perf = json.load(f)
        
        data[dataset] = {
            'statistics': stats,
            'performance': perf
        }
    
    return data


def plot_topk_comparison(data):
    """Comparison of top-k accuracy between datasets"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    datasets = []
    top_k_values = [1, 5, 10, 20, 50, 100, 500, 1000]
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        
        if 'top_k_accuracy' in stats:
            # COCO format
            accuracies = []
            for k in top_k_values:
                key = f'top_{k}'
                if key in stats['top_k_accuracy']:
                    acc_str = stats['top_k_accuracy'][key]
                    acc = float(acc_str.replace('%', ''))
                    accuracies.append(acc)
                else:
                    accuracies.append(None)
        else:
            # Flickr/VizWiz format
            accuracies = []
            for k in top_k_values:
                key = f'top_{k}_percentage'
                if key in stats:
                    accuracies.append(stats[key])
                else:
                    accuracies.append(None)
        
        datasets.append({
            'name': dataset,
            'accuracies': accuracies
        })
    
    # Plot
    x = np.arange(len(top_k_values))
    width = 0.25
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    for i, dataset_info in enumerate(datasets):
        offset = (i - 1) * width
        values = [v if v is not None else 0 for v in dataset_info['accuracies']]
        ax.bar(x + offset, values, width, label=dataset_info['name'], 
               color=colors[dataset_info['name']], alpha=0.8)
    
    ax.set_xlabel('Top-K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Top-K Accuracy entre Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_k_values)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_topk_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Top-K comparison saved")


def plot_rank_statistics(data):
    """Rank statistics (mean, median, std)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Data
    datasets_names = []
    means = []
    medians = []
    stds = []
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        datasets_names.append(dataset)
        
        if 'basic_statistics' in stats:
            means.append(stats['basic_statistics']['mean_rank'])
            medians.append(stats['basic_statistics']['median_rank'])
            stds.append(stats['basic_statistics']['std_rank'])
        else:
            means.append(stats.get('mean_rank', 0))
            medians.append(stats.get('median_rank', 0))
            stds.append(0)  # Pas de std dans Flickr/VizWiz
    
    # Plot 1: Mean vs Median
    x = np.arange(len(datasets_names))
    width = 0.35
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    bars1 = ax1.bar(x - width/2, means, width, label='Mean Rank', alpha=0.8)
    bars2 = ax1.bar(x + width/2, medians, width, label='Median Rank', alpha=0.8)
    
    # Colorer les barres par dataset
    for i, bar in enumerate(bars1):
        bar.set_color(colors[datasets_names[i]])
    for i, bar in enumerate(bars2):
        bar.set_color(colors[datasets_names[i]])
        bar.set_alpha(0.6)
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rank', fontsize=12, fontweight='bold')
    ax1.set_title('Mean vs Median Rank per Dataset', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Standard Deviation
    valid_datasets = [d for i, d in enumerate(datasets_names) if stds[i] > 0]
    valid_stds = [s for s in stds if s > 0]
    
    if valid_stds:
        bars = ax2.bar(valid_datasets, valid_stds, alpha=0.8)
        for i, bar in enumerate(bars):
            bar.set_color(colors[valid_datasets[i]])
        
        ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        ax2.set_title('Rank Standard Deviation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_rank_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Rank statistics saved")


def plot_percentiles_comparison(data):
    """Comparison of percentiles between datasets"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    percentile_keys = ['1th', '5th', '10th', '25th', '50th', '75th', '90th', '95th', '99th']
    percentile_labels = ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%']
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        
        if 'percentiles' in stats:
            values = [stats['percentiles'][key] for key in percentile_keys]
            colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
            ax.plot(percentile_labels, values, marker='o', linewidth=2.5, 
                   label=dataset, color=colors[dataset], markersize=8)
    
    ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rank', fontsize=12, fontweight='bold')
    ax.set_title('Rank Percentile Distribution per Dataset', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Logarithmic scale for better visibility
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_percentiles_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Percentiles comparison saved")


def plot_rank_distribution_boxplot(data):
    """Boxplot of rank distribution for each dataset"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    # Collect ranks for each dataset
    box_data = []
    box_labels = []
    box_colors = []
    
    for dataset in DATASETS:
        perf = data[dataset]['performance']
        ranks = []
        
        # Extract ranks from performance data
        if isinstance(perf, dict):
            for query_id, query_data in perf.items():
                if isinstance(query_data, dict) and 'rank' in query_data:
                    ranks.append(query_data['rank'])
                elif isinstance(query_data, (int, float)):
                    ranks.append(query_data)
        elif isinstance(perf, list):
            ranks = [item.get('rank', 0) if isinstance(item, dict) else item for item in perf]
        
        if ranks:
            box_data.append(ranks)
            box_labels.append(dataset)
            box_colors.append(colors[dataset])
    
    # Create boxplot
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    showfliers=True, widths=0.5,
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize whiskers, caps, and medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, alpha=0.7)
    for cap in bp['caps']:
        cap.set(linewidth=1.5, alpha=0.7)
    for median in bp['medians']:
        median.set(color='red', linewidth=2.5)
    
    # Add statistics annotations
    for i, (dataset, ranks) in enumerate(zip(box_labels, box_data)):
        stats = data[dataset]['statistics']
        if 'basic_statistics' in stats:
            median_val = stats['basic_statistics']['median_rank']
            mean_val = stats['basic_statistics']['mean_rank']
        else:
            median_val = stats.get('median_rank', np.median(ranks))
            mean_val = stats.get('mean_rank', np.mean(ranks))
        
        # Add text annotations
        ax.text(i+1, median_val, f'Median: {median_val:.1f}', 
               ha='left', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_ylabel('Rank (log scale)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Rank Distribution Boxplot Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sample size information
    info_text = ''
    for i, (dataset, ranks) in enumerate(zip(box_labels, box_data)):
        info_text += f'{dataset}: n={len(ranks):,}  '
    ax.text(0.5, -0.15, info_text, transform=ax.transAxes,
           ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_rank_distribution_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Rank distribution boxplot saved")


def plot_rank_density_distribution(data):
    """Density distribution of ranks for all three datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    for idx, dataset in enumerate(DATASETS):
        perf = data[dataset]['performance']
        ax = axes[idx]
        
        # Extract all ranks
        ranks = []
        if isinstance(perf, dict):
            for query_id, query_data in perf.items():
                if isinstance(query_data, dict) and 'rank' in query_data:
                    ranks.append(query_data['rank'])
                elif isinstance(query_data, (int, float)):
                    ranks.append(query_data)
        elif isinstance(perf, list):
            ranks = [item.get('rank', 0) if isinstance(item, dict) else item for item in perf]
        
        if not ranks:
            continue
        
        # Create many bins for detailed distribution
        max_rank = max(ranks)
        bins = 100  # 100 bins for detailed distribution
        
        # Plot histogram with density normalization
        ax.hist(ranks, bins=bins, color=colors[dataset], alpha=0.7, 
               edgecolor='black', linewidth=0.5, density=True)
        
        # Add statistics
        stats = data[dataset]['statistics']
        if 'basic_statistics' in stats:
            mean_val = stats['basic_statistics']['mean_rank']
            median_val = stats['basic_statistics']['median_rank']
        else:
            mean_val = stats.get('mean_rank', np.mean(ranks))
            median_val = stats.get('median_rank', np.median(ranks))
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        
        ax.set_xlabel('Rank', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset}\n(n={len(ranks):,})', fontsize=12, fontweight='bold', color=colors[dataset])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Rank Distribution Density by Dataset', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_rank_density_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Rank density distribution saved")


def plot_quality_distribution(data):
    """Results quality distribution - Cumulative table"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    # Prepare table data with cumulative percentages
    table_data = []
    headers = ['Dataset', '≤ 1\n(Perfect)', '≤ 10\n(+Excellent)', '≤ 100\n(+Good)', 
               '≤ 1000\n(+Fair)', 'All\n(+Poor)']
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        
        if 'quality_distribution' in stats:
            quality = stats['quality_distribution']
            perfect = quality['perfect']
            excellent = quality['excellent_2_10']
            good = quality['good_11_100']
            fair = quality['fair_101_1000']
            poor = quality['poor_1000_plus']
            total = perfect + excellent + good + fair + poor
        else:
            # For Flickr/VizWiz, create distribution based on top-k
            top_1 = stats.get('top_1_count', 0)
            top_10 = stats.get('top_10_count', 0)
            top_100 = stats.get('top_100_count', 0)
            top_1000 = stats.get('top_1000_count', 0)
            total = stats.get('total_queries', 0)
            
            perfect = top_1
            excellent = top_10 - top_1
            good = top_100 - top_10
            fair = top_1000 - top_100
            poor = total - top_1000
        
        # Calculate cumulative counts and percentages
        cum_1 = perfect
        cum_10 = perfect + excellent
        cum_100 = cum_10 + good
        cum_1000 = cum_100 + fair
        cum_all = total
        
        cum_1_str = f'{cum_1:,}\n({cum_1/total*100:.1f}%)'
        cum_10_str = f'{cum_10:,}\n({cum_10/total*100:.1f}%)'
        cum_100_str = f'{cum_100:,}\n({cum_100/total*100:.1f}%)'
        cum_1000_str = f'{cum_1000:,}\n({cum_1000/total*100:.1f}%)'
        cum_all_str = f'{cum_all:,}\n(100.0%)'
        
        table_data.append([dataset, cum_1_str, cum_10_str, cum_100_str, cum_1000_str, cum_all_str])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style rows
    quality_colors = ['#27ae60', '#2ecc71', '#95a5a6', '#e67e22', '#95a5a6']
    for i, dataset in enumerate(DATASETS):
        # Dataset column
        cell = table[(i+1, 0)]
        cell.set_facecolor(colors[dataset])
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        
        # Cumulative quality columns with color gradient
        for j in range(1, 6):
            cell = table[(i+1, j)]
            cell.set_facecolor(quality_colors[j-1])
            cell.set_text_props(color='white', fontsize=10, weight='bold')
            cell.set_alpha(0.7)
    
    plt.title('Cumulative Quality Distribution Table by Dataset', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / '04_quality_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Quality distribution saved")


def plot_cumulative_accuracy(data):
    """Cumulative accuracy curve"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    k_values = list(range(1, 101))  # From 1 to 100
    extended_k = [1, 5, 10, 20, 50, 100, 500, 1000]
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        
        # Interpolate values for smooth curve
        if 'top_k_accuracy' in stats:
            # COCO format
            known_k = []
            known_acc = []
            for k in extended_k:
                key = f'top_{k}'
                if key in stats['top_k_accuracy']:
                    known_k.append(k)
                    acc_str = stats['top_k_accuracy'][key]
                    known_acc.append(float(acc_str.replace('%', '')))
        else:
            # Flickr/VizWiz format
            known_k = []
            known_acc = []
            for k in extended_k:
                key = f'top_{k}_percentage'
                if key in stats:
                    known_k.append(k)
                    known_acc.append(stats[key])
        
        # Interpolation
        interpolated_acc = np.interp(k_values, known_k[:len(k_values)], known_acc[:len(k_values)])
        
        ax.plot(k_values, interpolated_acc, linewidth=2.5, label=dataset, 
               color=colors[dataset], marker='o', markersize=4, markevery=10)
    
    ax.set_xlabel('Top-K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Accuracy Curve (Top 1-100)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add horizontal reference lines
    for y in [25, 50, 75]:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_cumulative_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Cumulative accuracy saved")


def create_summary_table(data):
    """Create summary table of key metrics"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Dataset', 'Total Images', 'Mean Rank', 'Median Rank', 
              'Top-1 (%)', 'Top-10 (%)', 'Top-100 (%)']
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        
        # Total images
        if 'total_images' in stats:
            total = f"{stats['total_images']:,}"
        else:
            total = f"{stats.get('total_queries', 0):,}"
        
        # Mean rank
        if 'basic_statistics' in stats:
            mean_rank = f"{stats['basic_statistics']['mean_rank']:.2f}"
            median_rank = f"{stats['basic_statistics']['median_rank']:.1f}"
        else:
            mean_rank = f"{stats.get('mean_rank', 0):.2f}"
            median_rank = f"{stats.get('median_rank', 0):.1f}"
        
        # Top-K accuracies
        if 'top_k_accuracy' in stats:
            top1 = stats['top_k_accuracy'].get('top_1', 'N/A')
            top10 = stats['top_k_accuracy'].get('top_10', 'N/A')
            top100 = stats['top_k_accuracy'].get('top_100', 'N/A')
        else:
            top1 = f"{stats.get('top_1_percentage', 0):.2f}%"
            top10 = f"{stats.get('top_10_percentage', 0):.2f}%"
            top100 = f"{stats.get('top_100_percentage', 0):.2f}%"
        
        table_data.append([dataset, total, mean_rank, median_rank, top1, top10, top100])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    for i, dataset in enumerate(DATASETS):
        cell = table[(i+1, 0)]
        cell.set_facecolor(colors[dataset])
        cell.set_text_props(weight='bold', color='white')
        
        # Alternate colors for other cells
        for j in range(1, len(headers)):
            cell = table[(i+1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('#ffffff')
    
    plt.title('Performance Summary by Dataset', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / '06_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Summary table saved")


def plot_performance_heatmap(data):
    """Top-K performance heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    top_k_values = [1, 5, 10, 20, 50, 100, 500, 1000]
    matrix_data = []
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        row = []
        
        for k in top_k_values:
            if 'top_k_accuracy' in stats:
                key = f'top_{k}'
                if key in stats['top_k_accuracy']:
                    acc_str = stats['top_k_accuracy'][key]
                    acc = float(acc_str.replace('%', ''))
                    row.append(acc)
                else:
                    row.append(0)
            else:
                key = f'top_{k}_percentage'
                if key in stats:
                    row.append(stats[key])
                else:
                    row.append(0)
        
        matrix_data.append(row)
    
    # Create heatmap
    im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Configure axes
    ax.set_xticks(np.arange(len(top_k_values)))
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([f'Top-{k}' for k in top_k_values])
    ax.set_yticklabels(DATASETS)
    
    # Add values in cells
    for i in range(len(DATASETS)):
        for j in range(len(top_k_values)):
            text = ax.text(j, i, f'{matrix_data[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Top-K Performance Heatmap (%)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Performance heatmap saved")


def plot_dataset_comparison_radar(data):
    """Radar chart comparing datasets on different metrics"""
    from math import pi
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Metrics to compare (normalized 0-100)
    categories = ['Top-1', 'Top-10', 'Top-100', 'Median\n(inverted)', 'Top-1000']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    for dataset in DATASETS:
        stats = data[dataset]['statistics']
        
        # Extract values
        if 'top_k_accuracy' in stats:
            top1 = float(stats['top_k_accuracy'].get('top_1', '0').replace('%', ''))
            top10 = float(stats['top_k_accuracy'].get('top_10', '0').replace('%', ''))
            top100 = float(stats['top_k_accuracy'].get('top_100', '0').replace('%', ''))
            top1000 = float(stats['top_k_accuracy'].get('top_1000', '0').replace('%', ''))
            median = stats['basic_statistics']['median_rank']
        else:
            top1 = stats.get('top_1_percentage', 0)
            top10 = stats.get('top_10_percentage', 0)
            top100 = stats.get('top_100_percentage', 0)
            top1000 = stats.get('top_1000_percentage', 0)
            median = stats.get('median_rank', 0)
        
        # Normalize median (lower is better)
        # Invert so that 100 = best
        median_normalized = max(0, 100 - (median / 100) * 100)
        median_normalized = min(100, median_normalized)
        
        values = [top1, top10, top100, median_normalized, top1000]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=dataset, 
               color=colors[dataset], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[dataset])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('Multi-criteria Dataset Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Radar comparison saved")


def analyze_performance_details(data):
    """Analyze performance details from performance.json"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'COCO': '#3498db', 'Flickr': '#e74c3c', 'VizWiz': '#2ecc71'}
    
    # Get sample of ranks for each dataset
    for idx, dataset in enumerate(DATASETS):
        perf = data[dataset]['performance']
        
        # Extract ranks
        ranks = []
        if isinstance(perf, dict):
            for query_id, query_data in list(perf.items())[:5000]:  # Limit to 5000 for performance
                if isinstance(query_data, dict) and 'rank' in query_data:
                    ranks.append(query_data['rank'])
                elif isinstance(query_data, (int, float)):
                    ranks.append(query_data)
        elif isinstance(perf, list):
            ranks = [item.get('rank', 0) if isinstance(item, dict) else item for item in perf[:5000]]
        
        if not ranks:
            continue
        
        # Plot 1: Rank distribution (log scale with density)
        ax = axes[0, 0] if idx == 0 else axes[0, 0]
        if idx == 0:
            axes[0, 0].clear()
        
        # Histogram with density normalization
        bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, max(ranks)+1]
        axes[0, 0].hist(ranks, bins=bins, alpha=0.5, label=dataset, color=colors[dataset], 
                       edgecolor='black', density=True)
    
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Rank (log scale)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Rank Distribution Density (Logarithmic Scale)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Rank box plot
    box_data = []
    box_labels = []
    for dataset in DATASETS:
        perf = data[dataset]['performance']
        ranks = []
        if isinstance(perf, dict):
            for query_id, query_data in list(perf.items())[:5000]:
                if isinstance(query_data, dict) and 'rank' in query_data:
                    ranks.append(query_data['rank'])
                elif isinstance(query_data, (int, float)):
                    ranks.append(query_data)
        elif isinstance(perf, list):
            ranks = [item.get('rank', 0) if isinstance(item, dict) else item for item in perf[:5000]]
        
        if ranks:
            box_data.append(ranks)
            box_labels.append(dataset)
    
    bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True, 
                            showfliers=False, widths=0.6)
    
    for patch, dataset in zip(bp['boxes'], box_labels):
        patch.set_facecolor(colors[dataset])
        patch.set_alpha(0.7)
    
    axes[0, 1].set_ylabel('Rank', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Rank Box Plot (without outliers)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Percentage in different ranges
    tranche_labels = ['1', '2-10', '11-100', '101-1000', '>1000']
    tranche_data = {dataset: [] for dataset in DATASETS}
    
    for dataset in DATASETS:
        perf = data[dataset]['performance']
        ranks = []
        if isinstance(perf, dict):
            for query_id, query_data in perf.items():
                if isinstance(query_data, dict) and 'rank' in query_data:
                    ranks.append(query_data['rank'])
                elif isinstance(query_data, (int, float)):
                    ranks.append(query_data)
        elif isinstance(perf, list):
            ranks = [item.get('rank', 0) if isinstance(item, dict) else item for item in perf]
        
        if ranks:
            total = len(ranks)
            t1 = sum(1 for r in ranks if r == 1) / total * 100
            t2_10 = sum(1 for r in ranks if 2 <= r <= 10) / total * 100
            t11_100 = sum(1 for r in ranks if 11 <= r <= 100) / total * 100
            t101_1000 = sum(1 for r in ranks if 101 <= r <= 1000) / total * 100
            t1000_plus = sum(1 for r in ranks if r > 1000) / total * 100
            
            tranche_data[dataset] = [t1, t2_10, t11_100, t101_1000, t1000_plus]
    
    x = np.arange(len(tranche_labels))
    width = 0.25
    
    for i, dataset in enumerate(DATASETS):
        offset = (i - 1) * width
        axes[1, 0].bar(x + offset, tranche_data[dataset], width, 
                      label=dataset, color=colors[dataset], alpha=0.8)
    
    axes[1, 0].set_xlabel('Rank Range', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Distribution by Rank Ranges', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(tranche_labels)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative curve
    for dataset in DATASETS:
        perf = data[dataset]['performance']
        ranks = []
        if isinstance(perf, dict):
            for query_id, query_data in perf.items():
                if isinstance(query_data, dict) and 'rank' in query_data:
                    ranks.append(query_data['rank'])
                elif isinstance(query_data, (int, float)):
                    ranks.append(query_data)
        elif isinstance(perf, list):
            ranks = [item.get('rank', 0) if isinstance(item, dict) else item for item in perf]
        
        if ranks:
            sorted_ranks = sorted(ranks)
            cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks) * 100
            axes[1, 1].plot(sorted_ranks, cumulative, linewidth=2.5, 
                          label=dataset, color=colors[dataset])
    
    axes[1, 1].set_xlabel('Rank', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Cumulative Rank Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(1, 10000)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_detailed_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Detailed performance analysis saved")


def main():
    """Main function"""
    print("🚀 Starting performance analysis...")
    print(f"📁 Output directory: {OUTPUT_DIR}\n")
    
    # Load data
    print("📊 Loading data...")
    data = load_data()
    print("✓ Data loaded\n")
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    
    plot_topk_comparison(data)
    plot_rank_statistics(data)
    plot_percentiles_comparison(data)
    plot_quality_distribution(data)
    plot_cumulative_accuracy(data)
    create_summary_table(data)
    plot_performance_heatmap(data)
    plot_dataset_comparison_radar(data)
    analyze_performance_details(data)
    plot_rank_distribution_boxplot(data)
    plot_rank_density_distribution(data)
    
    print(f"\n✅ All visualizations generated in: {OUTPUT_DIR}")
    print(f"\n📈 Files created:")
    for i, name in enumerate([
        '01_topk_comparison.png',
        '02_rank_statistics.png',
        '03_percentiles_comparison.png',
        '04_quality_distribution.png',
        '05_cumulative_accuracy.png',
        '06_summary_table.png',
        '07_performance_heatmap.png',
        '08_radar_comparison.png',
        '09_detailed_performance_analysis.png',
        '10_rank_distribution_boxplot.png',
        '11_rank_density_distribution.png'
    ], 1):
        print(f"  {i}. {name}")


if __name__ == "__main__":
    main()
