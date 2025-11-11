#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Evaluation for Hybrid IVF Retrieval with LLM Cluster Summaries
Evaluates all fusion methods on the entire COCO dataset

Usage:
    # Evaluate all fusion methods on gemma3_4b
    python src/performance/performance_ivf_llm_desc.py --llm-name gemma3_4b --fusion-method all
    
    # Evaluate all fusion methods on all LLMs
    python src/performance/performance_ivf_llm_desc.py --all-llms --fusion-method all
    
    # Evaluate single fusion method
    python src/performance/performance_ivf_llm_desc.py --llm-name gemma3_4b --fusion-method rrf
    
    # Test on subset of images
    python src/performance/performance_ivf_llm_desc.py --llm-name gemma3_4b --fusion-method all --max-images 100
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever_ivf_llm_desc import HybridIVFRetriever


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """Evaluator for Hybrid IVF Retrieval performance"""
    
    def __init__(
        self,
        dataset: str = "COCO",
        llm_name: str = "gemma3_4b",
        fusion_method: str = "combsum",
        output_dir: Path = None,
        device: str = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """
        Args:
            dataset: Dataset name
            llm_name: LLM model name
            fusion_method: Fusion method to evaluate
            output_dir: Directory to save results
            device: Device for inference
        """
        self.dataset = dataset
        self.llm_name = llm_name
        self.fusion_method = fusion_method
        self.device = device
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # Output directory
        if output_dir is None:
            output_dir = project_root / "report" / "performance_ivf_llm_desc" / dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize retriever
        logger.info(f"Initializing retriever: dataset={dataset}, llm={llm_name}, fusion={fusion_method}")
        self.retriever = HybridIVFRetriever(
            dataset=dataset,
            llm_name=llm_name,
            fusion_method=fusion_method,
            device=device,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b
        )
        
        logger.info("✅ PerformanceEvaluator initialized")
    
    def evaluate_all_images(self, k_clusters: int = 10, max_images: int = None, save_frequency: int = 1) -> Dict:
        """
        Evaluate retrieval performance on all images
        
        Args:
            k_clusters: Number of top clusters to retrieve
            max_images: Maximum number of images to evaluate (None = all)
            save_frequency: Save results every N images (default: 1 = save after each image)
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting evaluation: {self.fusion_method.upper()}")
        logger.info(f"Dataset: {self.dataset}, LLM: {self.llm_name}")
        logger.info(f"k_clusters: {k_clusters}")
        logger.info(f"{'='*80}\n")
        
        # Get all filenames
        all_filenames = list(self.retriever.filename_to_caption.keys())
        
        if max_images:
            all_filenames = all_filenames[:max_images]
            logger.info(f"Evaluating on {len(all_filenames)} images (limited)")
        else:
            logger.info(f"Evaluating on {len(all_filenames)} images (full dataset)")
        
        # Prepare output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset}_ivf_hybrid_{self.llm_name}_{self.fusion_method}_{timestamp}.json"
        self.output_path = self.output_dir / filename
        
        logger.info(f"💾 Results will be saved to: {self.output_path}")
        if save_frequency > 0:
            logger.info(f"💾 Incremental saves every {save_frequency} images")
        
        # Results storage
        results = {
            "metadata": {
                "dataset": self.dataset,
                "llm_name": self.llm_name,
                "fusion_method": self.fusion_method,
                "k_clusters": k_clusters,
                "total_images": len(all_filenames),
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "save_frequency": save_frequency
            },
            "per_image_results": [],
            "statistics": {}
        }
        
        # Track statistics
        embedding_positions = []
        bm25_positions = []
        hybrid_positions = []
        
        embedding_times = []
        bm25_times = []
        fusion_times = []
        total_times = []
        
        found_in_topk_emb = 0
        found_in_topk_bm25 = 0
        found_in_topk_hybrid = 0
        
        errors = []
        
        # Process each image
        start_time = time.time()
        
        for idx, filename in enumerate(tqdm(all_filenames, desc=f"Evaluating {self.fusion_method}")):
            try:
                # Search (without verbose output - only show tqdm progress)
                result = self.retriever.search(
                    filename=filename,
                    k_clusters=k_clusters,
                    verbose=False
                )
                
                # Extract positions
                emb_pos = result["embedding_ranking"]["position"]
                bm25_pos = result["bm25_ranking"]["position"]
                hybrid_pos = result["hybrid_ranking"]["position"]
                
                # Collect statistics
                if emb_pos:
                    embedding_positions.append(emb_pos)
                    if emb_pos <= k_clusters:
                        found_in_topk_emb += 1
                
                if bm25_pos:
                    bm25_positions.append(bm25_pos)
                    if bm25_pos <= k_clusters:
                        found_in_topk_bm25 += 1
                
                if hybrid_pos:
                    hybrid_positions.append(hybrid_pos)
                    if hybrid_pos <= k_clusters:
                        found_in_topk_hybrid += 1
                
                # Collect timing
                embedding_times.append(result["embedding_ranking"]["time_seconds"])
                bm25_times.append(result["bm25_ranking"]["time_seconds"])
                fusion_times.append(result["hybrid_ranking"]["time_seconds"])
                total_times.append(result["total_time_seconds"])
                
                # Store result
                results["per_image_results"].append({
                    "filename": filename,
                    "caption": result["caption"][:100],  # Truncate
                    "real_cluster": result["real_cluster"],
                    "embedding_position": emb_pos,
                    "bm25_position": bm25_pos,
                    "hybrid_position": hybrid_pos,
                    "embedding_time_ms": result["embedding_ranking"]["time_seconds"] * 1000,
                    "bm25_time_ms": result["bm25_ranking"]["time_seconds"] * 1000,
                    "fusion_time_ms": result["hybrid_ranking"]["time_seconds"] * 1000,
                    "total_time_ms": result["total_time_seconds"] * 1000
                })
                
                # Incremental save
                if save_frequency > 0 and (idx + 1) % save_frequency == 0:
                    self._save_incremental(results, embedding_positions, bm25_positions, hybrid_positions,
                                          embedding_times, bm25_times, fusion_times, total_times,
                                          found_in_topk_emb, found_in_topk_bm25, found_in_topk_hybrid,
                                          errors, idx + 1)
                
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                logger.error(error_msg)
                errors.append({"filename": filename, "error": str(e)})
        
        total_eval_time = time.time() - start_time
        
        # Compute statistics
        n_total = len(all_filenames)
        n_errors = len(errors)
        n_success = n_total - n_errors
        
        results["statistics"] = {
            "total_images": n_total,
            "successful_retrievals": n_success,
            "errors": n_errors,
            "total_evaluation_time_seconds": total_eval_time,
            
            # Embedding ranking stats
            "embedding_ranking": {
                "found_in_topk": found_in_topk_emb,
                "recall_at_k": found_in_topk_emb / n_success if n_success > 0 else 0,
                "avg_position": float(np.mean(embedding_positions)) if embedding_positions else None,
                "median_position": float(np.median(embedding_positions)) if embedding_positions else None,
                "min_position": int(np.min(embedding_positions)) if embedding_positions else None,
                "max_position": int(np.max(embedding_positions)) if embedding_positions else None,
                "avg_time_ms": float(np.mean(embedding_times)) * 1000 if embedding_times else None,
            },
            
            # BM25 ranking stats
            "bm25_ranking": {
                "found_in_topk": found_in_topk_bm25,
                "recall_at_k": found_in_topk_bm25 / n_success if n_success > 0 else 0,
                "avg_position": float(np.mean(bm25_positions)) if bm25_positions else None,
                "median_position": float(np.median(bm25_positions)) if bm25_positions else None,
                "min_position": int(np.min(bm25_positions)) if bm25_positions else None,
                "max_position": int(np.max(bm25_positions)) if bm25_positions else None,
                "avg_time_ms": float(np.mean(bm25_times)) * 1000 if bm25_times else None,
            },
            
            # Hybrid ranking stats
            "hybrid_ranking": {
                "method": self.fusion_method,
                "found_in_topk": found_in_topk_hybrid,
                "recall_at_k": found_in_topk_hybrid / n_success if n_success > 0 else 0,
                "avg_position": float(np.mean(hybrid_positions)) if hybrid_positions else None,
                "median_position": float(np.median(hybrid_positions)) if hybrid_positions else None,
                "min_position": int(np.min(hybrid_positions)) if hybrid_positions else None,
                "max_position": int(np.max(hybrid_positions)) if hybrid_positions else None,
                "avg_time_ms": float(np.mean(fusion_times)) * 1000 if fusion_times else None,
            },
            
            # Total timing
            "timing": {
                "avg_total_time_ms": float(np.mean(total_times)) * 1000 if total_times else None,
                "median_total_time_ms": float(np.median(total_times)) * 1000 if total_times else None,
            }
        }
        
        results["errors"] = errors
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_incremental(self, results, embedding_positions, bm25_positions, hybrid_positions,
                         embedding_times, bm25_times, fusion_times, total_times,
                         found_in_topk_emb, found_in_topk_bm25, found_in_topk_hybrid,
                         errors, n_processed):
        """Save incremental results"""
        n_success = n_processed - len(errors)
        
        # Compute current statistics
        results["statistics"] = {
            "images_processed": n_processed,
            "successful_retrievals": n_success,
            "errors": len(errors),
            
            # Embedding ranking stats
            "embedding_ranking": {
                "found_in_topk": found_in_topk_emb,
                "recall_at_k": found_in_topk_emb / n_success if n_success > 0 else 0,
                "avg_position": float(np.mean(embedding_positions)) if embedding_positions else None,
                "median_position": float(np.median(embedding_positions)) if embedding_positions else None,
                "min_position": int(np.min(embedding_positions)) if embedding_positions else None,
                "max_position": int(np.max(embedding_positions)) if embedding_positions else None,
                "avg_time_ms": float(np.mean(embedding_times)) * 1000 if embedding_times else None,
            },
            
            # BM25 ranking stats
            "bm25_ranking": {
                "found_in_topk": found_in_topk_bm25,
                "recall_at_k": found_in_topk_bm25 / n_success if n_success > 0 else 0,
                "avg_position": float(np.mean(bm25_positions)) if bm25_positions else None,
                "median_position": float(np.median(bm25_positions)) if bm25_positions else None,
                "min_position": int(np.min(bm25_positions)) if bm25_positions else None,
                "max_position": int(np.max(bm25_positions)) if bm25_positions else None,
                "avg_time_ms": float(np.mean(bm25_times)) * 1000 if bm25_times else None,
            },
            
            # Hybrid ranking stats
            "hybrid_ranking": {
                "method": self.fusion_method,
                "found_in_topk": found_in_topk_hybrid,
                "recall_at_k": found_in_topk_hybrid / n_success if n_success > 0 else 0,
                "avg_position": float(np.mean(hybrid_positions)) if hybrid_positions else None,
                "median_position": float(np.median(hybrid_positions)) if hybrid_positions else None,
                "min_position": int(np.min(hybrid_positions)) if hybrid_positions else None,
                "max_position": int(np.max(hybrid_positions)) if hybrid_positions else None,
                "avg_time_ms": float(np.mean(fusion_times)) * 1000 if fusion_times else None,
            },
            
            # Total timing
            "timing": {
                "avg_total_time_ms": float(np.mean(total_times)) * 1000 if total_times else None,
                "median_total_time_ms": float(np.median(total_times)) * 1000 if total_times else None,
            }
        }
        
        results["errors"] = errors
        
        # Save to file
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Silent save - no log message
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        stats = results["statistics"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: {results['metadata']['dataset']}")
        logger.info(f"LLM: {results['metadata']['llm_name']}")
        logger.info(f"Fusion Method: {results['metadata']['fusion_method'].upper()}")
        logger.info(f"k_clusters: {results['metadata']['k_clusters']}")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Successful: {stats['successful_retrievals']}")
        logger.info(f"Errors: {stats['errors']}")
        
        logger.info(f"\n📊 EMBEDDING RANKING:")
        emb = stats["embedding_ranking"]
        logger.info(f"  Recall@k: {emb['recall_at_k']:.4f} ({emb['found_in_topk']}/{stats['successful_retrievals']})")
        logger.info(f"  Avg position: {emb['avg_position']:.2f}")
        logger.info(f"  Median position: {emb['median_position']:.0f}")
        logger.info(f"  Min/Max position: {emb['min_position']}/{emb['max_position']}")
        logger.info(f"  Avg time: {emb['avg_time_ms']:.2f} ms")
        
        logger.info(f"\n📊 BM25 RANKING:")
        bm25 = stats["bm25_ranking"]
        logger.info(f"  Recall@k: {bm25['recall_at_k']:.4f} ({bm25['found_in_topk']}/{stats['successful_retrievals']})")
        logger.info(f"  Avg position: {bm25['avg_position']:.2f}")
        logger.info(f"  Median position: {bm25['median_position']:.0f}")
        logger.info(f"  Min/Max position: {bm25['min_position']}/{bm25['max_position']}")
        logger.info(f"  Avg time: {bm25['avg_time_ms']:.2f} ms")
        
        logger.info(f"\n📊 HYBRID RANKING ({self.fusion_method.upper()}):")
        hyb = stats["hybrid_ranking"]
        logger.info(f"  Recall@k: {hyb['recall_at_k']:.4f} ({hyb['found_in_topk']}/{stats['successful_retrievals']})")
        logger.info(f"  Avg position: {hyb['avg_position']:.2f}")
        logger.info(f"  Median position: {hyb['median_position']:.0f}")
        logger.info(f"  Min/Max position: {hyb['min_position']}/{hyb['max_position']}")
        logger.info(f"  Avg time: {hyb['avg_time_ms']:.2f} ms")
        
        # Compare with embedding baseline
        if emb['recall_at_k'] > 0:
            improvement = hyb['recall_at_k'] - emb['recall_at_k']
            logger.info(f"\n📈 IMPROVEMENT vs EMBEDDING BASELINE:")
            logger.info(f"  Recall improvement: {improvement:+.4f} ({improvement/emb['recall_at_k']*100:+.2f}%)")
            
            if hyb['avg_position'] and emb['avg_position']:
                pos_improvement = emb['avg_position'] - hyb['avg_position']
                logger.info(f"  Avg position improvement: {pos_improvement:+.2f}")
        
        logger.info(f"\n⏱️  TOTAL TIMING:")
        timing = stats["timing"]
        logger.info(f"  Avg total time: {timing['avg_total_time_ms']:.2f} ms")
        logger.info(f"  Median total time: {timing['median_total_time_ms']:.2f} ms")
        logger.info(f"  Total evaluation time: {stats['total_evaluation_time_seconds']:.2f} seconds")
        
        logger.info(f"\n{'='*80}\n")
    
    def save_results(self, results: Dict):
        """Save final results to JSON file"""
        logger.info(f"💾 Saving final results to: {self.output_path}")
        
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Results saved!")
        
        # Also save a summary file (can be overwritten)
        summary_filename = f"{self.dataset}_ivf_hybrid_{self.llm_name}_{self.fusion_method}_latest.json"
        summary_path = self.output_dir / summary_filename
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Latest results also saved to: {summary_path}")
        
        return self.output_path


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ce code doit remplacer la fonction evaluate_all_fusion_methods 
dans src/performance/performance_ivf_llm_desc.py

Instructions:
1. Ouvrir src/performance/performance_ivf_llm_desc.py
2. Trouver la fonction evaluate_all_fusion_methods (ligne ~429)
3. Remplacer TOUTE la fonction par le code ci-dessous
"""

# ============= CODE À COPIER =============

def evaluate_all_fusion_methods(
    dataset: str = "COCO",
    llm_name: str = "gemma3_4b",
    k_clusters: int = 10,
    max_images: int = None,
    device: str = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    save_frequency: int = 1
):
    """Evaluate all fusion methods and save in a SINGLE JSON file"""
    fusion_methods = ["combsum", "borda", "max_pooling", "rrf", 
                      "weighted_combsum", "combmnz", "zscore_combsum"]
    
    logger.info(f"EVALUATING ALL FUSION METHODS")
    logger.info(f"Dataset: {dataset}, LLM: {llm_name}")
    
    # Créer une structure pour stocker TOUTES les méthodes dans un seul JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "metadata": {
            "dataset": dataset,
            "llm_name": llm_name,
            "k_clusters": k_clusters,
            "max_images": max_images,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "timestamp": timestamp,
            "fusion_methods": fusion_methods
        },
        "methods": {}
    }
    
    # Déterminer output_dir
    project_root = Path.cwd()
    output_dir = project_root / "report" / "performance_ivf_llm_desc" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # UN SEUL fichier pour TOUTES les méthodes
    output_file = output_dir / f"{dataset}_ivf_hybrid_{llm_name}_all_methods_{timestamp}.json"
    logger.info(f"📁 Results will be saved to: {output_file}")
    
    for idx, fusion_method in enumerate(fusion_methods, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating fusion method {idx}/{len(fusion_methods)}: {fusion_method}")
        logger.info(f"{'='*80}\n")
        
        try:
            evaluator = PerformanceEvaluator(
                dataset=dataset,
                llm_name=llm_name,
                fusion_method=fusion_method,
                output_dir=output_dir,
                device=device,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b
            )
            
            results = evaluator.evaluate_all_images(
                k_clusters=k_clusters,
                max_images=max_images,
                save_frequency=save_frequency
            )
            
            # Ajouter les résultats de cette méthode à la structure combinée
            all_results["methods"][fusion_method] = results
            
            # Sauvegarder après CHAQUE méthode (incrémental)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Completed {fusion_method} ({idx}/{len(fusion_methods)})")
            logger.info(f"   Top-1: {results['summary']['hybrid_ranking']['top1_accuracy']:.2%}")
            logger.info(f"   Top-5: {results['summary']['hybrid_ranking']['top5_accuracy']:.2%}")
            logger.info(f"   Top-10: {results['summary']['hybrid_ranking']['top10_accuracy']:.2%}")
            logger.info(f"💾 Saved progress to: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {fusion_method}: {e}")
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ ALL {len(fusion_methods)} EVALUATIONS COMPLETE")
    logger.info(f"📁 Final results in: {output_file}")
    logger.info(f"{'='*80}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Performance Evaluation for Hybrid IVF Retrieval"
    )
    parser.add_argument("--dataset", type=str, default="COCO", help="Dataset name")
    parser.add_argument("--llm-name", type=str, default="gemma3_4b",
                       help="LLM model name (gemma3_4b, mistral_7b)")
    parser.add_argument("--fusion-method", type=str, default=None,
                       choices=["combsum", "borda", "max_pooling", "rrf", 
                               "weighted_combsum", "combmnz", "zscore_combsum", "all"],
                       help="Fusion method to evaluate (or 'all' for all methods)")
    parser.add_argument("--k-clusters", type=int, default=10,
                       help="Number of top clusters to retrieve")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to evaluate (None = all)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for inference (cuda/cpu)")
    parser.add_argument("--bm25-k1", type=float, default=1.5,
                       help="BM25 k1 parameter (term frequency saturation, 1.2-2.0)")
    parser.add_argument("--bm25-b", type=float, default=0.75,
                       help="BM25 b parameter (document length normalization, 0-1)")
    parser.add_argument("--all-llms", action="store_true",
                       help="Evaluate on all available LLMs")
    parser.add_argument("--save-frequency", type=int, default=1,
                       help="Save results every N images (default: 1 = after each image, 0 = only at end)")
    
    args = parser.parse_args()
    
    # Determine which LLMs to evaluate
    if args.all_llms:
        llm_names = ["gemma3_4b", "mistral_7b", "gemma3_27b"]
    else:
        llm_names = [args.llm_name]
    
    # Evaluate each LLM
    for llm_name in llm_names:
        logger.info(f"\n{'#'*80}")
        logger.info(f"EVALUATING LLM: {llm_name.upper()}")
        logger.info(f"{'#'*80}\n")
        
        if args.fusion_method == "all" or args.fusion_method is None:
            # Evaluate all fusion methods
            evaluate_all_fusion_methods(
                dataset=args.dataset,
                llm_name=llm_name,
                k_clusters=args.k_clusters,
                max_images=args.max_images,
                device=args.device,
                save_frequency=args.save_frequency
            )
        else:
            # Evaluate single fusion method
            evaluator = PerformanceEvaluator(
                dataset=args.dataset,
                llm_name=llm_name,
                fusion_method=args.fusion_method,
                device=args.device
            )
            
            results = evaluator.evaluate_all_images(
                k_clusters=args.k_clusters,
                max_images=args.max_images,
                save_frequency=args.save_frequency
            )
            
            evaluator.save_results(results)
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"✅ EVALUATION COMPLETE!")
    logger.info(f"{'#'*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
