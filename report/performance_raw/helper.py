#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inject baseline retrieval ranks into multiple dataset metadata files.

For each dataset:
- Read performance.json, expecting a list under --ranks-key (default: 'ranks').
- Read metadata.json, expecting a list under --images-key (default: 'images').
- For i in range(min(len(images), len(ranks))):
    images[i][--field] = ranks[i]
- Writes back in-place with a .bak backup, unless --out-dir is provided.

Examples
--------
# Run on explicit pairs (repeat --pair for multiple datasets)
python update_baseline_rank.py \
  --pair "C:\\...\\report\\performance_raw\\COCO\\performance.json|C:\\...\\data\\COCO\\coco_metadata.json" \
  --pair "C:\\...\\report\\performance_raw\\Flickr\\performance.json|C:\\...\\data\\Flickr\\flickr_metadata.json" \
  --pair "C:\\...\\report\\performance_raw\\VizWiz\\performance.json|C:\\...\\data\\VizWiz\\VizWiz_metadata.json"

# Run with built-in defaults for COCO, Flickr, VizWiz
python update_baseline_rank.py --use-defaults
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import statistics
from typing import List, Tuple, Dict, Any

def update_metadata(
    perf_path: Path,
    meta_path: Path,
    ranks_key: str = "ranks",
    images_key: str = "images",
    field_name: str = "baseline_rank",
    out_path: Path | None = None,
) -> None:
    # Load performance.json
    try:
        with perf_path.open("r", encoding="utf-8") as f:
            perf = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Performance file not found: {perf_path}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {perf_path}: {e}", file=sys.stderr)
        return

    ranks = perf.get(ranks_key)
    if not isinstance(ranks, list):
        print(f"[ERROR] '{ranks_key}' must be a list in {perf_path}", file=sys.stderr)
        return

    # Load metadata.json
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Metadata file not found: {meta_path}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {meta_path}: {e}", file=sys.stderr)
        return

    images = meta.get(images_key)
    if not isinstance(images, list):
        print(f"[ERROR] '{images_key}' must be a list in {meta_path}", file=sys.stderr)
        return

    n = min(len(images), len(ranks))
    if n == 0:
        print(f"[WARN] No updates (empty '{images_key}' or '{ranks_key}') for {meta_path.name}", file=sys.stderr)
        return

    # Apply updates
    for i in range(n):
        if isinstance(images[i], dict):
            images[i][field_name] = ranks[i]
        else:
            print(f"[WARN] {meta_path.name}: images[{i}] is not an object; skipped", file=sys.stderr)

    if len(images) != len(ranks):
        print(
            f"[WARN] Size mismatch for {meta_path.name}: {images_key}={len(images)} vs {ranks_key}={len(ranks)}; "
            f"updated first {n} records.",
            file=sys.stderr,
        )

    # Write output (with backup if in-place)
    target = out_path if out_path else meta_path
    if target == meta_path:
        backup = meta_path.with_suffix(meta_path.suffix + ".bak")
        shutil.copy2(meta_path, backup)
        print(f"[INFO] Backup created: {backup}")

    with target.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {field_name} for {n} items → {target}")

def parse_pair_arg(pair_arg: str) -> Tuple[Path, Path]:
    """
    Parse a pair string like:
      "C:\\path\\to\\performance.json|C:\\path\\to\\metadata.json"
    """
    if "|" not in pair_arg:
        raise ValueError("Pair must be 'PERFORMANCE_PATH|METADATA_PATH'")
    perf, meta = pair_arg.split("|", 1)
    return Path(perf), Path(meta)

def default_pairs() -> List[Tuple[Path, Path]]:
    # Your defaults (Windows paths)
    pairs = [
        (
            Path(r"C:\Users\binbi\Desktop\DataLab2Project\report\performance_raw\COCO\performance.json"),
            Path(r"C:\Users\binbi\Desktop\DataLab2Project\data\COCO\coco_metadata.json"),
        ),
        (
            Path(r"C:\Users\binbi\Desktop\DataLab2Project\report\performance_raw\Flickr\performance.json"),
            Path(r"C:\Users\binbi\Desktop\DataLab2Project\data\Flickr\flickr_metadata.json"),
        ),
        (
            Path(r"C:\Users\binbi\Desktop\DataLab2Project\report\performance_raw\VizWiz\performance.json"),
            Path(r"C:\Users\binbi\Desktop\DataLab2Project\data\VizWiz\VizWiz_metadata.json"),
        ),
    ]
    return pairs

def compute_rank_statistics(ranks: List[int]) -> Dict[str, Any]:
    """
    Compute ranking statistics from a list of ranks.
    
    Args:
        ranks: List of ranking positions (1-indexed)
        
    Returns:
        Dictionary containing top-k percentages and basic statistics
    """
    if not ranks:
        return {}
    
    total_count = len(ranks)
    
    # Calculate top-k percentages
    top_1 = sum(1 for rank in ranks if rank == 1)
    top_2 = sum(1 for rank in ranks if rank <= 2)
    top_3 = sum(1 for rank in ranks if rank <= 3)
    top_4 = sum(1 for rank in ranks if rank <= 4)
    top_5 = sum(1 for rank in ranks if rank <= 5)
    top_6 = sum(1 for rank in ranks if rank <= 6)
    top_7 = sum(1 for rank in ranks if rank <= 7)
    top_8 = sum(1 for rank in ranks if rank <= 8)
    top_9 = sum(1 for rank in ranks if rank <= 9)
    top_10 = sum(1 for rank in ranks if rank <= 10)
    top_100 = sum(1 for rank in ranks if rank <= 100)
    top_1000 = sum(1 for rank in ranks if rank <= 1000)
    
    statistics_dict = {
        "total_queries": total_count,
        "top_1_count": top_1,
        "top_2_count": top_2,
        "top_3_count": top_3,
        "top_4_count": top_4,
        "top_5_count": top_5,
        "top_6_count": top_6,
        "top_7_count": top_7,
        "top_8_count": top_8,
        "top_9_count": top_9,
        "top_10_count": top_10,
        "top_100_count": top_100,
        "top_1000_count": top_1000,
        "top_1_percentage": round((top_1 / total_count) * 100, 2),
        "top_2_percentage": round((top_2 / total_count) * 100, 2),
        "top_3_percentage": round((top_3 / total_count) * 100, 2),
        "top_4_percentage": round((top_4 / total_count) * 100, 2),
        "top_5_percentage": round((top_5 / total_count) * 100, 2),
        "top_6_percentage": round((top_6 / total_count) * 100, 2),
        "top_7_percentage": round((top_7 / total_count) * 100, 2),
        "top_8_percentage": round((top_8 / total_count) * 100, 2),
        "top_9_percentage": round((top_9 / total_count) * 100, 2),
        "top_10_percentage": round((top_10 / total_count) * 100, 2),
        "top_100_percentage": round((top_100 / total_count) * 100, 2),
        "top_1000_percentage": round((top_1000 / total_count) * 100, 2),
        "mean_rank": round(statistics.mean(ranks), 2),
        "median_rank": round(statistics.median(ranks), 2),
        "min_rank": min(ranks),
        "max_rank": max(ranks)
    }
    
    return statistics_dict

def compute_and_save_statistics(dataset_name: str) -> None:
    """
    Compute statistics for a specific dataset and save them to statistics.json.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'Flickr', 'VizWiz', 'COCO')
    """
    perf_path = Path(f"C:\\Users\\binbi\\Desktop\\DataLab2Project\\report\\performance_raw\\{dataset_name}\\performance.json")
    stats_path = Path(f"C:\\Users\\binbi\\Desktop\\DataLab2Project\\report\\performance_raw\\{dataset_name}\\statistics.json")
    
    # Load performance.json
    try:
        with perf_path.open("r", encoding="utf-8") as f:
            perf_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Performance file not found: {perf_path}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {perf_path}: {e}", file=sys.stderr)
        return
    
    ranks = perf_data.get("ranks")
    if not isinstance(ranks, list):
        print(f"[ERROR] 'ranks' must be a list in {perf_path}", file=sys.stderr)
        return
    
    if not ranks:
        print(f"[WARN] Empty ranks list in {perf_path}", file=sys.stderr)
        return
    
    # Compute statistics
    stats = compute_rank_statistics(ranks)
    
    # Add metadata from original performance file
    stats["dataset_name"] = dataset_name
    stats["vectordb_name"] = perf_data.get("vectordb_name", "")
    stats["evaluation_date"] = perf_data.get("evaluation_date", "")
    stats["total_images_in_db"] = perf_data.get("total_images_in_db", 0)
    stats["images_evaluated"] = perf_data.get("images_evaluated", 0)
    stats["computation_date"] = json.loads(json.dumps(perf_data.get("evaluation_date", "")))  # Copy date format
    
    # Save statistics.json
    try:
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[OK] Statistics saved to {stats_path}")
        
        # Print summary
        print(f"\n=== {dataset_name} Statistics ===")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Top-1: {stats['top_1_percentage']}% ({stats['top_1_count']}/{stats['total_queries']})")
        print(f"Top-2: {stats['top_2_percentage']}% ({stats['top_2_count']}/{stats['total_queries']})")
        print(f"Top-3: {stats['top_3_percentage']}% ({stats['top_3_count']}/{stats['total_queries']})")
        print(f"Top-4: {stats['top_4_percentage']}% ({stats['top_4_count']}/{stats['total_queries']})")
        print(f"Top-5: {stats['top_5_percentage']}% ({stats['top_5_count']}/{stats['total_queries']})")
        print(f"Top-6: {stats['top_6_percentage']}% ({stats['top_6_count']}/{stats['total_queries']})")
        print(f"Top-7: {stats['top_7_percentage']}% ({stats['top_7_count']}/{stats['total_queries']})")
        print(f"Top-8: {stats['top_8_percentage']}% ({stats['top_8_count']}/{stats['total_queries']})")
        print(f"Top-9: {stats['top_9_percentage']}% ({stats['top_9_count']}/{stats['total_queries']})")
        print(f"Top-10: {stats['top_10_percentage']}% ({stats['top_10_count']}/{stats['total_queries']})")
        print(f"Top-100: {stats['top_100_percentage']}% ({stats['top_100_count']}/{stats['total_queries']})")
        print(f"Top-1000: {stats['top_1000_percentage']}% ({stats['top_1000_count']}/{stats['total_queries']})")
        print(f"Mean rank: {stats['mean_rank']}")
        print(f"Median rank: {stats['median_rank']}")
        
    except Exception as e:
        print(f"[ERROR] Could not save statistics to {stats_path}: {e}", file=sys.stderr)

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inject baseline retrieval ranks into dataset metadata files or compute statistics."
    )
    
    # Mutual exclusive group for different operations
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pair",
        action="append",
        help="Pair of PERFORMANCE_PATH|METADATA_PATH (can be repeated)"
    )
    group.add_argument(
        "--use-defaults",
        action="store_true",
        help="Use built-in default paths for COCO, Flickr, VizWiz"
    )
    group.add_argument(
        "--compute-stats",
        action="store_true",
        help="Compute and save statistics for Flickr and VizWiz datasets"
    )
    
    # Optional arguments
    parser.add_argument("--ranks-key", default="ranks", help="Key for ranks in performance.json")
    parser.add_argument("--images-key", default="images", help="Key for images in metadata.json")
    parser.add_argument("--field", default="baseline_rank", help="Field name to add to metadata")
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: in-place)")
    
    args = parser.parse_args()
    
    if args.compute_stats:
        # Compute statistics for Flickr and VizWiz
        for dataset in ["Flickr", "VizWiz"]:
            compute_and_save_statistics(dataset)
        return
    
    # Handle pair operations (existing functionality)
    pairs = []
    if args.use_defaults:
        pairs = default_pairs()
    elif args.pair:
        for pair_str in args.pair:
            try:
                pairs.append(parse_pair_arg(pair_str))
            except ValueError as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                sys.exit(1)
    
    for perf_path, meta_path in pairs:
        out_path = None
        if args.out_dir:
            out_path = args.out_dir / meta_path.name
        
        update_metadata(
            perf_path,
            meta_path,
            args.ranks_key,
            args.images_key,
            args.field,
            out_path,
        )

if __name__ == "__main__":
    main()
