#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Evaluation for IVF Hybrid Retrieval with LLM Cluster Summaries
Evaluates the hybrid retrieval combining CLIP embeddings with BM25 on LLM-generated cluster summaries

Usage:
python src/performance/performance_ivf_llm_desc.py --llm-name gemma3_4b --fusion-method combsum
python src/performance/performance_ivf_llm_desc.py --llm-name all --fusion-method all
"""

import json
import sys
import argparse
import time
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

# Import the hybrid retriever
sys.path.append(str(Path(__file__).parent.parent))
from retrieval.retriever_ivf_llm_desc import HybridIVFRetriever
from config.config import Config


class HybridIVFPerformanceEvaluator:
    """
    Performance evaluator for hybrid IVF retrieval combining CLIP embeddings and BM25
    """
    
    def __init__(
        self,
        dataset: str = "COCO",
        llm_name: str = "gemma3_4b",
        fusion_method: str = "combsum",
        bm25_k1: float = None,
        bm25_b: float = None,
        use_sample: bool = True,
        output_file: Optional[str] = None
    ):
        """
        Args:
            dataset: Dataset name (e.g., "COCO")
            llm_name: LLM model name for cluster summaries
            fusion_method: Fusion method (combsum, borda, rrf)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            use_sample: Whether to use 10k sample (True) or full dataset (False)
            output_file: Optional output file path for results
        """
        self.dataset = dataset
        self.use_sample = use_sample
        self.llm_name = llm_name
        self.fusion_method = fusion_method
        self.data_dir = Config.get_dataset_dir(dataset)
        self.report_dir = Config.REPORT_DIR / "performance_ivf_llm_desc" / dataset
        self.output_file = output_file
        
        # Create report directory
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Load baseline cluster positions
        if self.use_sample:
            # Use existing cluster_positions.json for sample evaluation
            self.cluster_positions_path = self.data_dir / "cluster_positions.json"
            if not self.cluster_positions_path.exists():
                raise FileNotFoundError(f"Baseline cluster positions not found: {self.cluster_positions_path}")
        else:
            # Use full COCO metadata for complete dataset evaluation
            self.cluster_positions_path = self.data_dir / "coco_metadata.json"
            if not self.cluster_positions_path.exists():
                raise FileNotFoundError(f"COCO metadata not found: {self.cluster_positions_path}")
        
        with open(self.cluster_positions_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Adapt data structure based on file type
        if self.use_sample:
            # cluster_positions.json structure: {"results": [...]}
            self.baseline_data = raw_data
        else:
            # coco_metadata.json structure: {"images": [...]}
            # Convert to expected structure
            self.baseline_data = {
                "results": [
                    {
                        "filename": img["filename"],
                        "cluster_position": img.get("baseline_rank", None),
                        "image_id": img.get("image_id", None)
                    }
                    for img in raw_data.get("images", [])
                    if img.get("baseline_rank") is not None
                ]
            }
        
        eval_type = "sample" if self.use_sample else "full dataset"
        print(f"✅ Loaded baseline data ({eval_type}) with {len(self.baseline_data.get('results', []))} results")
        
        # BM25 parameters
        self.bm25_k1 = bm25_k1 if bm25_k1 is not None else Config.BM25_K1
        self.bm25_b = bm25_b if bm25_b is not None else Config.BM25_B
    
    def load_existing_results(self, output_path: Path) -> Dict:
        """
        Load existing evaluation results for resume functionality
        
        Args:
            output_path: Path to the results file
        
        Returns:
            Existing results dictionary or empty structure
        """
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                print(f"📂 Found existing results file: {output_path}")
                existing_methods = list(existing_results.get('methods', {}).keys())
                print(f"   Existing methods: {existing_methods}")
                return existing_results
            except Exception as e:
                print(f"⚠️  Failed to load existing results: {e}")
                print(f"   Starting fresh evaluation...")
        
        return None
    
    def get_pending_combinations(self, existing_results: Dict, llm_names: List[str], fusion_methods: List[str]) -> List[Tuple[str, str]]:
        """
        Get list of LLM/fusion combinations that still need evaluation
        
        Args:
            existing_results: Existing results dictionary
            llm_names: All LLM names to evaluate
            fusion_methods: All fusion methods to evaluate
        
        Returns:
            List of (llm_name, fusion_method) tuples pending evaluation
        """
        all_combinations = [(llm, fusion) for llm in llm_names for fusion in fusion_methods]
        
        if not existing_results:
            return all_combinations
        
        existing_methods = set(existing_results.get("methods", {}).keys())
        pending_combinations = []
        
        for llm_name, fusion_method in all_combinations:
            method_key = f"{llm_name}_{fusion_method}"
            if method_key not in existing_methods:
                pending_combinations.append((llm_name, fusion_method))
        
        return pending_combinations
    
    def _save_intermediate_results(
        self, 
        llm_name: str, 
        fusion_method: str, 
        results: List[Dict], 
        sampled_results: List[Dict],
        successful: int,
        failed: int
    ):
        """
        Save intermediate results during evaluation
        
        Args:
            llm_name: Current LLM being evaluated
            fusion_method: Current fusion method being evaluated
            results: Current results list
            sampled_results: Original sampled results
            successful: Number of successful evaluations so far
            failed: Number of failed evaluations so far
        """
        try:
            # Create intermediate results structure
            intermediate_data = {
                "llm_name": llm_name,
                "fusion_method": fusion_method,
                "dataset": self.dataset,
                "progress": {
                    "total_images": len(sampled_results),
                    "processed": len(results),
                    "successful": successful,
                    "failed": failed,
                    "progress_percent": (len(results) / len(sampled_results)) * 100
                },
                "bm25_parameters": {
                    "k1": self.bm25_k1,
                    "b": self.bm25_b
                },
                "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "results": results
            }
            
            # Save to intermediate file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_filename = f"COCO_hybrid_ivf_llm_{llm_name}_{fusion_method}_progress_{len(results)}_of_{len(sampled_results)}.json"
            intermediate_path = self.report_dir / "intermediate" / intermediate_filename
            
            # Create intermediate directory if it doesn't exist
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Intermediate progress saved: {len(results)}/{len(sampled_results)} images")
            
        except Exception as e:
            print(f"⚠️  Failed to save intermediate results: {e}")
        
    def sample_evaluation_images(
        self, 
        n_samples: int = 10000,
        min_position: int = 2,
        top_percentile: float = 0.9
    ) -> List[Dict]:
        """
        Sample evaluation images or return full dataset based on use_sample parameter
        
        Args:
            n_samples: Number of images to sample (ignored if use_sample=False)
            min_position: Minimum cluster position (default: 2, ignored if use_sample=False)
            top_percentile: Top percentile to include (default: 0.9, ignored if use_sample=False)
        
        Returns:
            List of sampled image results or full dataset results
        """
        if not self.use_sample:
            # Return all images from the full COCO dataset
            all_results = self.baseline_data.get('results', [])
            print(f"\n🎯 Using full COCO dataset: {len(all_results)} images")
            # Filter out images without baseline_rank/cluster_position
            valid_results = [r for r in all_results if r.get('cluster_position') is not None]
            print(f"   Valid images with baseline position: {len(valid_results)}")
            return valid_results
        
        print(f"\n🎯 Sampling {n_samples} evaluation images...")
        print(f"   Criteria: position >= {min_position} and in top {top_percentile*100:.0f}%")
        
        # Get all baseline results
        all_results = self.baseline_data.get("results", [])
        
        # Filter results by criteria
        valid_results = []
        positions = []
        
        for result in all_results:
            pos = result.get("cluster_position")
            if pos is not None and pos >= min_position:
                valid_results.append(result)
                positions.append(pos)
        
        if not positions:
            raise ValueError(f"No results found with position >= {min_position}")
        
        # Calculate top percentile threshold
        positions_sorted = sorted(positions)
        threshold_idx = int(len(positions_sorted) * top_percentile)
        position_threshold = positions_sorted[threshold_idx] if threshold_idx < len(positions_sorted) else max(positions_sorted)
        
        # Filter by top percentile
        filtered_results = [
            result for result in valid_results
            if result.get("cluster_position", float('inf')) <= position_threshold
        ]
        
        print(f"   Found {len(filtered_results)} images meeting criteria")
        print(f"   Position range: {min_position} - {position_threshold}")
        
        if len(filtered_results) < n_samples:
            print(f"⚠️  Only {len(filtered_results)} images available, using all")
            sampled_results = filtered_results
        else:
            # Sample uniformly across position ranges for balanced distribution
            position_bins = {}
            for result in filtered_results:
                pos = result.get("cluster_position")
                bin_key = pos // 10  # Group by tens
                if bin_key not in position_bins:
                    position_bins[bin_key] = []
                position_bins[bin_key].append(result)
            
            # Sample proportionally from each bin
            sampled_results = []
            bin_sizes = {k: len(v) for k, v in position_bins.items()}
            total_in_bins = sum(bin_sizes.values())
            
            for bin_key, bin_results in position_bins.items():
                bin_proportion = len(bin_results) / total_in_bins
                bin_samples = max(1, int(n_samples * bin_proportion))
                bin_samples = min(bin_samples, len(bin_results))
                
                sampled_from_bin = random.sample(bin_results, bin_samples)
                sampled_results.extend(sampled_from_bin)
            
            # Trim to exact number if we went over
            if len(sampled_results) > n_samples:
                sampled_results = random.sample(sampled_results, n_samples)
        
        print(f"✅ Sampled {len(sampled_results)} images")
        
        # Save sampled dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.use_sample:
            sample_filename = f"evaluation_sample_{n_samples}_pos{min_position}+_top{int(top_percentile*100)}p_{timestamp}.json"
            sample_criteria = f"position >= {min_position} and <= {position_threshold} (top {top_percentile*100:.0f}%)"
            position_threshold_val = position_threshold
        else:
            sample_filename = f"evaluation_full_dataset_{len(sampled_results)}_images_{timestamp}.json"
            sample_criteria = "Full COCO dataset - no filtering"
            position_threshold_val = None
        
        sample_path = self.data_dir / sample_filename
        
        sample_data = {
            "metadata": {
                "dataset": self.dataset,
                "evaluation_type": "sample" if self.use_sample else "full_dataset",
                "n_samples": len(sampled_results),
                "min_position": min_position if self.use_sample else None,
                "top_percentile": top_percentile if self.use_sample else None,
                "position_threshold": position_threshold_val,
                "timestamp": timestamp,
                "sample_criteria": sample_criteria
            },
            "sampled_images": sampled_results
        }
        
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        eval_type = "sampled dataset" if self.use_sample else "full dataset"
        print(f"💾 Saved {eval_type} to: {sample_path}")
        
        return sampled_results
    
    def evaluate_single_image(self, image_result: Dict, retriever: HybridIVFRetriever) -> Dict:
        """
        Evaluate a single image with the hybrid retriever
        
        Args:
            image_result: Baseline result for the image
            retriever: Hybrid retriever instance
        
        Returns:
            Evaluation result dictionary
        """
        filename = image_result.get("filename")
        baseline_position = image_result.get("cluster_position")
        
        if not filename:
            return {
                "filename": "unknown",
                "baseline_position": baseline_position,
                "error": "Missing filename in image_result",
                "success": False
            }
        
        if baseline_position is None:
            return {
                "filename": filename,
                "baseline_position": None,
                "error": "Missing baseline position",
                "success": False
            }
        
        try:
            # Run hybrid retrieval
            result = retriever.search(filename, k_clusters=10, verbose=False)
            
            # Extract positions
            emb_position = result["embedding_ranking"]["position"]
            bm25_position = result["bm25_ranking"]["position"]  
            hybrid_position = result["hybrid_ranking"]["position"]
            
            # Calculate improvements
            emb_improvement = baseline_position - emb_position if (baseline_position and emb_position) else None
            hybrid_improvement = baseline_position - hybrid_position if (baseline_position and hybrid_position) else None
            
            return {
                "filename": filename,
                "baseline_position": baseline_position,
                "embedding_position": emb_position,
                "bm25_position": bm25_position,
                "hybrid_position": hybrid_position,
                "embedding_improvement": emb_improvement,
                "hybrid_improvement": hybrid_improvement,
                "search_time_ms": result["total_time_seconds"] * 1000,
                "fusion_method": result["hybrid_ranking"]["method"],
                "success": True
            }
            
        except Exception as e:
            return {
                "filename": filename,
                "baseline_position": baseline_position,
                "error": str(e),
                "success": False
            }
    
    def evaluate_all_images(
        self, 
        sampled_results: List[Dict],
        llm_name: str,
        fusion_method: str
    ) -> Dict:
        """
        Evaluate all sampled images with specific LLM and fusion method
        
        Args:
            sampled_results: List of sampled image results
            llm_name: LLM model name
            fusion_method: Fusion method
        
        Returns:
            Complete evaluation results
        """
        print(f"\n🔬 Evaluating {len(sampled_results)} images...")
        print(f"   LLM: {llm_name}")
        print(f"   Fusion: {fusion_method}")
        
        # Initialize retriever
        try:
            retriever = HybridIVFRetriever(
                dataset=self.dataset,
                llm_name=llm_name,
                fusion_method=fusion_method,
                device="cuda",
                bm25_k1=self.bm25_k1,
                bm25_b=self.bm25_b
            )
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            return {"error": str(e), "results": []}
        
        # Evaluate each image with detailed progress tracking
        results = []
        successful = 0
        failed = 0
        
        # Create progress bar with ETA
        pbar = tqdm(
            sampled_results, 
            desc=f"🔬 {llm_name}-{fusion_method}",
            unit="img",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for i, image_result in enumerate(pbar):
            eval_result = self.evaluate_single_image(image_result, retriever)
            results.append(eval_result)
            
            if eval_result["success"]:
                successful += 1
            else:
                failed += 1
            
            # Update progress bar with current statistics
            success_rate = (successful / (i + 1)) * 100
            pbar.set_postfix({
                'Success': f'{success_rate:.1f}%',
                'OK': successful,
                'Fail': failed
            })
            
            # Save progress every 100 images
            if (i + 1) % 100 == 0:
                self._save_intermediate_results(llm_name, fusion_method, results, sampled_results, successful, failed)
        
        pbar.close()
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            # Position statistics
            baseline_positions = [r["baseline_position"] for r in successful_results if r["baseline_position"]]
            embedding_positions = [r["embedding_position"] for r in successful_results if r["embedding_position"]]
            hybrid_positions = [r["hybrid_position"] for r in successful_results if r["hybrid_position"]]
            
            # Improvement statistics
            emb_improvements = [r["embedding_improvement"] for r in successful_results if r["embedding_improvement"] is not None]
            hybrid_improvements = [r["hybrid_improvement"] for r in successful_results if r["hybrid_improvement"] is not None]
            
            # Performance metrics
            emb_better = len([i for i in emb_improvements if i > 0])
            emb_worse = len([i for i in emb_improvements if i < 0])
            emb_same = len([i for i in emb_improvements if i == 0])
            
            hybrid_better = len([i for i in hybrid_improvements if i > 0])
            hybrid_worse = len([i for i in hybrid_improvements if i < 0])
            hybrid_same = len([i for i in hybrid_improvements if i == 0])
            
            statistics = {
                "total_evaluated": len(successful_results),
                "baseline_stats": {
                    "mean": np.mean(baseline_positions) if baseline_positions else None,
                    "median": np.median(baseline_positions) if baseline_positions else None,
                    "min": min(baseline_positions) if baseline_positions else None,
                    "max": max(baseline_positions) if baseline_positions else None
                },
                "embedding_stats": {
                    "mean": np.mean(embedding_positions) if embedding_positions else None,
                    "median": np.median(embedding_positions) if embedding_positions else None,
                    "min": min(embedding_positions) if embedding_positions else None,
                    "max": max(embedding_positions) if embedding_positions else None
                },
                "hybrid_stats": {
                    "mean": np.mean(hybrid_positions) if hybrid_positions else None,
                    "median": np.median(hybrid_positions) if hybrid_positions else None,
                    "min": min(hybrid_positions) if hybrid_positions else None,
                    "max": max(hybrid_positions) if hybrid_positions else None
                },
                "embedding_improvements": {
                    "mean": np.mean(emb_improvements) if emb_improvements else None,
                    "median": np.median(emb_improvements) if emb_improvements else None,
                    "better": emb_better,
                    "worse": emb_worse,
                    "same": emb_same,
                    "better_rate": emb_better / len(emb_improvements) if emb_improvements else 0
                },
                "hybrid_improvements": {
                    "mean": np.mean(hybrid_improvements) if hybrid_improvements else None,
                    "median": np.median(hybrid_improvements) if hybrid_improvements else None,
                    "better": hybrid_better,
                    "worse": hybrid_worse,
                    "same": hybrid_same,
                    "better_rate": hybrid_better / len(hybrid_improvements) if hybrid_improvements else 0
                }
            }
            
        else:
            statistics = {"error": "No successful evaluations"}
        
        # Summary
        evaluation_result = {
            "llm_name": llm_name,
            "fusion_method": fusion_method,
            "dataset": self.dataset,
            "bm25_parameters": {
                "k1": self.bm25_k1,
                "b": self.bm25_b
            },
            "evaluation_summary": {
                "total_images": len(sampled_results),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(sampled_results) if sampled_results else 0
            },
            "statistics": statistics,
            "results": results
        }
        
        print(f"✅ Evaluation complete: {successful}/{len(sampled_results)} successful")
        if successful_results and hybrid_improvements:
            print(f"   Hybrid improvement rate: {hybrid_better}/{len(hybrid_improvements)} = {hybrid_better/len(hybrid_improvements)*100:.1f}%")
            print(f"   Average improvement: {np.mean(hybrid_improvements):.2f} positions")
        
        return evaluation_result
    
    def evaluate_all_fusion_methods(
        self, 
        sampled_results: List[Dict],
        llm_names: List[str] = None,
        fusion_methods: List[str] = None
    ) -> Dict:
        """
        Evaluate all LLM models and fusion methods with resume capability
        
        Args:
            sampled_results: List of sampled image results
            llm_names: List of LLM models to evaluate
            fusion_methods: List of fusion methods to evaluate
        
        Returns:
            Complete evaluation results for all combinations
        """
        if llm_names is None:
            llm_names = ["mistral_7b", "gemma3_4b", "gemma3_27b"]
        
        if fusion_methods is None:
            fusion_methods = ["combsum", "borda", "rrf"]
        
        # Determine output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_file:
            output_path = Path(self.output_file)
        else:
            output_filename = f"COCO_hybrid_ivf_llm_evaluation_all_methods_{timestamp}.json"
            output_path = self.report_dir / output_filename
        
        # Try to load existing results for resume
        existing_results = self.load_existing_results(output_path)
        
        # Initialize or update results structure
        if existing_results:
            combined_results = existing_results
            # Update metadata with current run info
            combined_results["metadata"].update({
                "last_updated": timestamp,
                "resumed": True
            })
        else:
            combined_results = {
                "metadata": {
                    "dataset": self.dataset,
                    "evaluation_type": "hybrid_ivf_llm_bm25",
                    "dataset_scope": "sample" if self.use_sample else "full_dataset",
                    "timestamp": timestamp,
                    "n_images": len(sampled_results),
                    "llm_models": llm_names,
                    "fusion_methods": fusion_methods,
                    "bm25_parameters": {
                        "k1": self.bm25_k1,
                        "b": self.bm25_b
                    },
                    "resumed": False
                },
                "methods": {}
            }
        
        # Get pending combinations (for resume functionality)
        pending_combinations = self.get_pending_combinations(existing_results, llm_names, fusion_methods)
        total_combinations = len(llm_names) * len(fusion_methods)
        completed_combinations = total_combinations - len(pending_combinations)
        
        if pending_combinations:
            print(f"\n🚀 {'Resuming' if existing_results else 'Starting'} comprehensive evaluation...")
            print(f"   LLM models: {llm_names}")
            print(f"   Fusion methods: {fusion_methods}")
            print(f"   Total combinations: {total_combinations}")
            if existing_results:
                print(f"   Already completed: {completed_combinations}")
                print(f"   Remaining: {len(pending_combinations)}")
            print(f"   Output file: {output_path}")
        else:
            print(f"\n✅ All combinations already evaluated!")
            print(f"   Results file: {output_path}")
            return combined_results
        
        # Create overall progress bar for all combinations
        overall_pbar = tqdm(
            total=total_combinations,
            desc="🎯 Overall Progress",
            initial=completed_combinations,
            unit="combination",
            position=0,
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc} {postfix}"
        )
        
        try:
            # Evaluate pending combinations
            for i, (llm_name, fusion_method) in enumerate(pending_combinations):
                method_key = f"{llm_name}_{fusion_method}"
                current_global = completed_combinations + i + 1
                
                overall_pbar.set_postfix({'Current': method_key})
                
                print(f"\n{'='*80}")
                print(f"🔬 Evaluation {current_global}/{total_combinations}: {method_key}")
                print(f"{'='*80}")
                
                try:
                    # Evaluate this combination
                    method_results = self.evaluate_all_images(sampled_results, llm_name, fusion_method)
                    combined_results["methods"][method_key] = method_results
                    
                    # Save incrementally after each successful evaluation
                    combined_results["metadata"]["last_completed"] = method_key
                    combined_results["metadata"]["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(combined_results, f, indent=2, ensure_ascii=False)
                    
                    print(f"💾 Progress saved: {method_key} completed")
                    
                except Exception as e:
                    print(f"❌ Failed to evaluate {method_key}: {e}")
                    # Save error info but continue
                    combined_results["methods"][method_key] = {
                        "error": str(e),
                        "failed_at": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(combined_results, f, indent=2, ensure_ascii=False)
                
                overall_pbar.update(1)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Evaluation interrupted by user")
            print(f"💾 Progress saved to: {output_path}")
            print(f"🔄 You can resume by running the same command again")
            return combined_results
        
        finally:
            overall_pbar.close()
        
        # Mark as completed
        combined_results["metadata"]["completed"] = True
        combined_results["metadata"]["completion_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Final save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"🎉 EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"📊 Final results saved to: {output_path}")
        
        # Summary statistics
        print(f"\n📈 SUMMARY:")
        successful_methods = 0
        failed_methods = 0
        
        for method_key, method_result in combined_results["methods"].items():
            if "error" in method_result:
                print(f"   ❌ {method_key}: Evaluation failed - {method_result['error'][:50]}...")
                failed_methods += 1
            else:
                stats = method_result.get("statistics", {})
                hybrid_stats = stats.get("hybrid_improvements", {})
                
                if hybrid_stats and "better_rate" in hybrid_stats:
                    improvement_rate = hybrid_stats["better_rate"] * 100
                    mean_improvement = hybrid_stats.get("mean", 0)
                    print(f"   ✅ {method_key}: {improvement_rate:.1f}% improvement rate, avg: {mean_improvement:.2f} positions")
                    successful_methods += 1
                else:
                    print(f"   ⚠️  {method_key}: Evaluation incomplete")
                    failed_methods += 1
        
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Successful evaluations: {successful_methods}/{total_combinations}")
        print(f"   Failed evaluations: {failed_methods}/{total_combinations}")
        print(f"   Success rate: {(successful_methods/total_combinations)*100:.1f}%")
        
        return combined_results


def main():
    parser = argparse.ArgumentParser(description="Hybrid IVF Performance Evaluation")
    parser.add_argument("--dataset", type=str, default="COCO", help="Dataset name")
    parser.add_argument("--llm-name", type=str, default="gemma3_4b", 
                       choices=["mistral_7b", "gemma3_4b", "gemma3_27b", "all"],
                       help="LLM model name or 'all' for all models")
    parser.add_argument("--fusion-method", type=str, default="combsum",
                       choices=["combsum", "borda", "rrf", "all"],
                       help="Fusion method or 'all' for all methods")
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of images to sample for evaluation")
    parser.add_argument("--min-position", type=int, default=2,
                       help="Minimum baseline cluster position")
    parser.add_argument("--top-percentile", type=float, default=0.9,
                       help="Top percentile to include (0.9 = top 90%)")
    parser.add_argument("--use-sample", action="store_true", default=True,
                       help="Use 10k sample for evaluation (default: True)")
    parser.add_argument("--use-full-dataset", action="store_true", default=False,
                       help="Use full COCO dataset for evaluation (overrides --use-sample)")
    parser.add_argument("--bm25-k1", type=float, default=None,
                       help="BM25 k1 parameter (default: from Config)")
    parser.add_argument("--bm25-b", type=float, default=None,
                       help="BM25 b parameter (default: from Config)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Determine evaluation scope
    use_sample = not args.use_full_dataset  # use_full_dataset overrides use_sample
    
    # Initialize evaluator
    evaluator = HybridIVFPerformanceEvaluator(
        dataset=args.dataset,
        llm_name=args.llm_name if args.llm_name != "all" else "gemma3_4b",
        fusion_method=args.fusion_method if args.fusion_method != "all" else "combsum",
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        use_sample=use_sample,
        output_file=args.output_file
    )
    
    # Sample evaluation images
    sampled_results = evaluator.sample_evaluation_images(
        n_samples=args.n_samples,
        min_position=args.min_position,
        top_percentile=args.top_percentile
    )
    
    # Determine evaluation scope
    if args.llm_name == "all" and args.fusion_method == "all":
        # Evaluate all combinations
        results = evaluator.evaluate_all_fusion_methods(sampled_results)
    elif args.llm_name == "all":
        # All LLMs, single fusion method
        llm_names = ["mistral_7b", "gemma3_4b", "gemma3_27b"]
        results = evaluator.evaluate_all_fusion_methods(
            sampled_results, 
            llm_names=llm_names, 
            fusion_methods=[args.fusion_method]
        )
    elif args.fusion_method == "all":
        # Single LLM, all fusion methods
        fusion_methods = ["combsum", "borda", "rrf"]
        results = evaluator.evaluate_all_fusion_methods(
            sampled_results,
            llm_names=[args.llm_name],
            fusion_methods=fusion_methods
        )
    else:
        # Single combination
        results = evaluator.evaluate_all_images(sampled_results, args.llm_name, args.fusion_method)
        
        # Save single result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            output_filename = f"COCO_hybrid_ivf_llm_{args.llm_name}_{args.fusion_method}_{timestamp}.json"
            output_path = evaluator.report_dir / output_filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ Evaluation completed successfully!")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
