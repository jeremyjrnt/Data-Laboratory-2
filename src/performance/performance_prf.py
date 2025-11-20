#!/usr/bin/env python3
"""
Performance Evaluation for Pseudo-Relevance Feedback (PRF) Retriever
Evaluates different LLM models with various Rocchio parameters on image retrieval.



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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retriever_prf import ImageRetriever
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformancePRFEvaluator:
    """
    Evaluates PRF retrieval performance across different LLMs and Rocchio parameters.
    """

    def __init__(self, dataset_name: str):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Name of the dataset (e.g., 'COCO', 'Flickr', 'VizWiz')
        """
        self.dataset_name = dataset_name
        self.vectordb_name = f"{dataset_name}_VectorDB"

        # LLM models to test
        self.llm_models = [
            "gemma3:4b",
            "gemma3:27b"
        ]

        # Rocchio parameters to test
        # Format: (alpha, beta, gamma)
        self.rocchio_params = [
            (1.0, 0.75, 0.0),  # Default/Moderate feedback
        ]

        # Setup directories
        self.output_dir = Config.REPORT_DIR / "performance_prf" / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Single consolidated results file for this dataset
        self.results_file = self.output_dir / f"performance_prf_{dataset_name}.json"

        # Load selected test cases
        self.selected_images = self._load_selected_images()

        logger.info(f"📊 Performance PRF Evaluator initialized for {dataset_name}")
        logger.info(f"🖼️  Test images: {len(self.selected_images)}")
        logger.info(f"🤖 LLM models: {self.llm_models}")
        logger.info(f"⚙️  Rocchio params: {len(self.rocchio_params)} configurations")

    def _load_selected_images(self) -> List[Dict]:
        """Load the selected_1000.json file for the dataset."""
        selected_path = Config.get_selected_images_path(self.dataset_name)

        if not selected_path.exists():
            raise FileNotFoundError(f"Selected images file not found: {selected_path}")

        with open(selected_path, 'r', encoding='utf-8') as f:
            images = json.load(f)

        logger.info(f"📂 Loaded {len(images)} images from {selected_path}")
        return images

    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use in filenames (Windows-compatible)."""
        invalid_chars = [':', '*', '?', '"', '<', '>', '|', '\\', '/']
        sanitized = model_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized

    def _get_output_filename(self, llm_model: str, alpha: float, beta: float, gamma: float) -> str:
        """Generate output filename for a specific LLM and Rocchio parameters."""
        sanitized_model = self._sanitize_model_name(llm_model)
        return f"performance_{sanitized_model}_a{alpha}_b{beta}_g{gamma}.json"

    def evaluate_single_configuration(
        self,
        llm_model: str,
        alpha: float,
        beta: float,
        gamma: float,
        k: int = 10
    ) -> Dict:
        """
        Evaluate a single LLM model with specific Rocchio parameters.

        Args:
            llm_model: LLM model name
            alpha: Rocchio alpha parameter (original query weight)
            beta: Rocchio beta parameter (relevant docs weight)
            gamma: Rocchio gamma parameter (non-relevant docs weight)
            k: Number of results to retrieve

        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("=" * 80)
        logger.info(f"🎯 Evaluating: {llm_model} with α={alpha}, β={beta}, γ={gamma}")
        logger.info(f"🤖 LLM Model: {llm_model}")
        logger.info(f"📊 Processing {len(self.selected_images)} images")
        logger.info("=" * 80)

        start_time = time.time()

        # Initialize retriever
        try:
            retriever = ImageRetriever(
                vectordb_name=self.vectordb_name,
                vectordb_dir="VectorDBs",
                llm_model=llm_model,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize retriever: {e}")
            return None

        # Load existing results if available to resume where we left off
        config_key = f"alpha_{alpha}_beta_{beta}_gamma_{gamma}"
        existing_results = None
        already_processed_filenames = set()
        
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                    if llm_model in all_results.get('llm_results', {}) and \
                       config_key in all_results['llm_results'][llm_model]:
                        existing_results = all_results['llm_results'][llm_model][config_key]
                        already_processed_filenames = {
                            img['filename'] for img in existing_results.get('image_results', [])
                        }
                        logger.info(f"📂 Found {len(already_processed_filenames)} already processed images, resuming...")
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing results: {e}")

        # Results storage - use existing results or create new
        if existing_results:
            results = existing_results
            # Update metadata
            results['metadata']['last_updated'] = datetime.now().isoformat()
            results['metadata']['resumed'] = True
        else:
            results = {
                'metadata': {
                    'dataset': self.dataset_name,
                    'llm_model': llm_model,
                    'rocchio_params': {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma
                    },
                    'k': k,
                    'num_images': len(self.selected_images),
                    'evaluation_date': datetime.now().isoformat(),
                },
                'image_results': []
            }

        # Process each image
        for idx, image_info in enumerate(self.selected_images, 1):
            filename = image_info['filename']
            image_id = image_info.get('image_id', filename)
            caption = image_info.get('caption', '')
            baseline_rank = image_info.get('baseline_rank')

            # Skip if already processed
            if filename in already_processed_filenames:
                logger.info(f"\n[{idx}/{len(self.selected_images)}] Skipping (already processed): {filename}")
                continue

            logger.info(f"\n[{idx}/{len(self.selected_images)}] Processing: {filename}")

            try:
                # Retrieve with PRF
                # Note: The retriever_prf.py uses default Rocchio params internally
                # We would need to modify it to accept custom params
                # For now, using the default behavior
                retrieval_result = retriever.retrieve_from_image(filename, k=k)

                if retrieval_result:
                    # Extract key information
                    target_tracking = retrieval_result.get('target_tracking', {})
                    llm_eval = retrieval_result.get('llm_evaluation', {})
                    prf_data = retrieval_result.get('prf_data', {})

                    # Store result
                    image_result = {
                        'image_id': image_id,
                        'filename': filename,
                        'caption': caption,
                        'baseline_rank': baseline_rank,
                        'baseline_rank_from_metadata': target_tracking.get('baseline_rank'),
                        'baseline_similarity': target_tracking.get('baseline_similarity'),
                        'in_baseline_topk': target_tracking.get('in_baseline_topk'),
                        'baseline_topk_rank': target_tracking.get('baseline_topk_rank'),
                        'llm_evaluation': {
                            'relevant_indices': llm_eval.get('relevant_indices', []),
                            'num_relevant': len(llm_eval.get('relevant_indices', [])),
                            'reasoning': llm_eval.get('reasoning', '')
                        },
                        'prf_rank': target_tracking.get('prf_rank'),
                        'prf_similarity': target_tracking.get('prf_similarity'),
                        'rank_improvement': target_tracking.get('improvement'),
                        'baseline_top_k': [
                            {
                                'rank': r['rank'],
                                'filename': r['filename'],
                                'similarity': r['similarity'],
                                'blip_description': r.get('blip_description', '')
                            }
                            for r in retrieval_result.get('baseline_results', [])
                        ],
                        'prf_top_k': [
                            {
                                'prf_rank': r['prf_rank'],
                                'filename': r['filename'],
                                'similarity': r.get('similarity', 0.0),
                                'blip_description': r.get('blip_description', ''),
                                'is_target': r.get('is_target', False)
                            }
                            for r in prf_data.get('prf_top_k', [])
                        ] if prf_data else []
                    }

                    results['image_results'].append(image_result)

                    # Log progress
                    if target_tracking.get('improvement') is not None:
                        improvement = target_tracking['improvement']
                        if improvement > 0:
                            logger.info(f"✅ Improvement: +{improvement} positions")
                        elif improvement < 0:
                            logger.info(f"⚠️  Degradation: {improvement} positions")
                        else:
                            logger.info(f"➡️  No change in rank")
                    else:
                        logger.info(f"ℹ️  Target not found in PRF results")

                else:
                    logger.warning(f"⚠️  No retrieval result for {filename}")
                    # Store failed result
                    results['image_results'].append({
                        'image_id': image_id,
                        'filename': filename,
                        'caption': caption,
                        'baseline_rank': baseline_rank,
                        'error': 'Retrieval failed'
                    })

            except Exception as e:
                logger.error(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                # Store error result
                results['image_results'].append({
                    'image_id': image_id,
                    'filename': filename,
                    'caption': caption,
                    'baseline_rank': baseline_rank,
                    'error': str(e)
                })

            # Save progress periodically (every 50 images)
            if idx % 50 == 0:
                self._save_results(results, llm_model, alpha, beta, gamma)
                logger.info(f"💾 Progress saved ({idx}/{len(self.selected_images)})")
            
            # Also save after each image (incremental saving)
            if idx % 1 == 0:
                self._save_results(results, llm_model, alpha, beta, gamma)

        # Final save
        self._save_results(results, llm_model, alpha, beta, gamma)

        elapsed_time = time.time() - start_time
        results['metadata']['elapsed_time_seconds'] = elapsed_time

        logger.info("=" * 80)
        logger.info(f"✅ Evaluation complete: {llm_model} (α={alpha}, β={beta}, γ={gamma})")
        logger.info(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
        logger.info("=" * 80)

        return results

    def _save_results(
        self,
        results: Dict,
        llm_model: str,
        alpha: float,
        beta: float,
        gamma: float
    ):
        """Save results to the consolidated JSON file."""
        # Load existing results if file exists
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = {
                'dataset': self.dataset_name,
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'llm_results': {}
            }
        
        # Update last_updated timestamp
        all_results['last_updated'] = datetime.now().isoformat()
        
        # Create key for this LLM and configuration
        config_key = f"alpha_{alpha}_beta_{beta}_gamma_{gamma}"
        
        # Initialize LLM entry if not exists
        if llm_model not in all_results['llm_results']:
            all_results['llm_results'][llm_model] = {}
        
        # Store results for this configuration
        all_results['llm_results'][llm_model][config_key] = results
        
        # Save consolidated file
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to {self.results_file}")
    
    def _update_consolidated_results(
        self,
        results: Dict,
        llm_model: str,
        alpha: float,
        beta: float,
        gamma: float
    ):
        """Update the consolidated results file with new results."""
        # Load existing consolidated results if file exists
        if self.consolidated_results_path.exists():
            with open(self.consolidated_results_path, 'r', encoding='utf-8') as f:
                consolidated = json.load(f)
        else:
            consolidated = {
                'dataset': self.dataset_name,
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'llm_results': {}
            }
        
        # Update last_updated timestamp
        consolidated['last_updated'] = datetime.now().isoformat()
        
        # Create key for this LLM and configuration
        config_key = f"alpha_{alpha}_beta_{beta}_gamma_{gamma}"
        
        # Initialize LLM entry if not exists
        if llm_model not in consolidated['llm_results']:
            consolidated['llm_results'][llm_model] = {}
        
        # Store results for this configuration
        consolidated['llm_results'][llm_model][config_key] = results
        
        # Save consolidated file
        with open(self.consolidated_results_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📦 Consolidated results updated: {self.consolidated_results_path}")
        
        # Also update global consolidated file (all datasets)
        self._update_global_consolidated(results, llm_model, alpha, beta, gamma)
    
    def _update_global_consolidated(
        self,
        results: Dict,
        llm_model: str,
        alpha: float,
        beta: float,
        gamma: float
    ):
        """Update the global consolidated results file (all datasets and LLMs)."""
        # Create parent directory if it doesn't exist
        self.global_consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing global results if file exists
        if self.global_consolidated_path.exists():
            with open(self.global_consolidated_path, 'r', encoding='utf-8') as f:
                global_results = json.load(f)
        else:
            global_results = {
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'datasets': {}
            }
        
        # Update last_updated timestamp
        global_results['last_updated'] = datetime.now().isoformat()
        
        # Initialize dataset entry if not exists
        if self.dataset_name not in global_results['datasets']:
            global_results['datasets'][self.dataset_name] = {
                'llm_results': {}
            }
        
        # Create key for this configuration
        config_key = f"alpha_{alpha}_beta_{beta}_gamma_{gamma}"
        
        # Initialize LLM entry if not exists
        if llm_model not in global_results['datasets'][self.dataset_name]['llm_results']:
            global_results['datasets'][self.dataset_name]['llm_results'][llm_model] = {}
        
        # Store results for this configuration
        global_results['datasets'][self.dataset_name]['llm_results'][llm_model][config_key] = results
        
        # Save global consolidated file
        with open(self.global_consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(global_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🌍 Global consolidated results updated: {self.global_consolidated_path}")



    def _compute_statistics(self, results: Dict) -> Dict:
        """Compute statistics from results."""
        image_results = results.get('image_results', [])

        # Filter out error cases
        valid_results = [r for r in image_results if 'error' not in r]

        if not valid_results:
            return {
                'total_images': len(image_results),
                'valid_images': 0,
                'error_images': len(image_results)
            }

        # Count improvements, degradations, no changes
        improvements = [r for r in valid_results if r.get('rank_improvement', 0) > 0]
        degradations = [r for r in valid_results if r.get('rank_improvement', 0) < 0]
        no_changes = [r for r in valid_results if r.get('rank_improvement', 0) == 0]
        not_found = [r for r in valid_results if r.get('prf_rank') is None]

        # LLM evaluation stats
        llm_found_relevant = [r for r in valid_results if r.get('llm_evaluation', {}).get('num_relevant', 0) > 0]

        stats = {
            'total_images': len(image_results),
            'valid_images': len(valid_results),
            'error_images': len(image_results) - len(valid_results),
            'improvements': {
                'count': len(improvements),
                'percentage': len(improvements) / len(valid_results) * 100 if valid_results else 0,
                'avg_improvement': sum(r['rank_improvement'] for r in improvements) / len(improvements) if improvements else 0
            },
            'degradations': {
                'count': len(degradations),
                'percentage': len(degradations) / len(valid_results) * 100 if valid_results else 0,
                'avg_degradation': sum(r['rank_improvement'] for r in degradations) / len(degradations) if degradations else 0
            },
            'no_changes': {
                'count': len(no_changes),
                'percentage': len(no_changes) / len(valid_results) * 100 if valid_results else 0
            },
            'not_found_in_prf': {
                'count': len(not_found),
                'percentage': len(not_found) / len(valid_results) * 100 if valid_results else 0
            },
            'llm_found_relevant': {
                'count': len(llm_found_relevant),
                'percentage': len(llm_found_relevant) / len(valid_results) * 100 if valid_results else 0,
                'avg_relevant_per_image': sum(r.get('llm_evaluation', {}).get('num_relevant', 0) for r in valid_results) / len(valid_results) if valid_results else 0
            }
        }

        return stats

    def evaluate_all_configurations(self, k: int = 10):
        """
        Evaluate all LLM models with all Rocchio parameter configurations.

        Args:
            k: Number of results to retrieve
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING COMPREHENSIVE PRF EVALUATION")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset: {self.dataset_name}")
        logger.info(f"🖼️  Images: {len(self.selected_images)}")
        logger.info(f"🤖 LLM Models: {len(self.llm_models)}")
        logger.info(f"⚙️  Rocchio Configs: {len(self.rocchio_params)}")
        logger.info(f"📈 Total Evaluations: {len(self.llm_models) * len(self.rocchio_params)}")
        logger.info("=" * 80 + "\n")

        total_evaluations = len(self.llm_models) * len(self.rocchio_params)
        current_eval = 0

        all_results = []

        for llm_model in self.llm_models:
            for alpha, beta, gamma in self.rocchio_params:
                current_eval += 1

                logger.info(f"\n{'=' * 80}")
                logger.info(f"📊 Evaluation {current_eval}/{total_evaluations}")
                logger.info(f"{'=' * 80}\n")

                results = self.evaluate_single_configuration(
                    llm_model=llm_model,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    k=k
                )

                if results:
                    # Compute and add statistics
                    stats = self._compute_statistics(results)
                    results['statistics'] = stats

                    # Save with statistics
                    self._save_results(results, llm_model, alpha, beta, gamma)

                    all_results.append({
                        'llm_model': llm_model,
                        'rocchio_params': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
                        'statistics': stats
                    })

                    # Display statistics
                    logger.info("\n📊 Statistics:")
                    logger.info(f"  ✅ Improvements: {stats['improvements']['count']} ({stats['improvements']['percentage']:.2f}%)")
                    logger.info(f"  ⚠️  Degradations: {stats['degradations']['count']} ({stats['degradations']['percentage']:.2f}%)")
                    logger.info(f"  ➡️  No changes: {stats['no_changes']['count']} ({stats['no_changes']['percentage']:.2f}%)")
                    logger.info(f"  🤖 LLM found relevant: {stats['llm_found_relevant']['count']} ({stats['llm_found_relevant']['percentage']:.2f}%)")

        # Save summary of all results
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': self.dataset_name,
                'evaluation_date': datetime.now().isoformat(),
                'total_evaluations': total_evaluations,
                'results': all_results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ Summary saved to {summary_path}")
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ALL EVALUATIONS COMPLETE")
        logger.info("=" * 80)


def evaluate_all_datasets(k: int = 10):
    """
    Execute performance evaluation on all three datasets (COCO, Flickr, VizWiz) sequentially.
    
    Args:
        k: Number of results to retrieve for each evaluation
    """
    datasets = ['COCO', 'Flickr', 'VizWiz']
    
    logger.info("\n" + "=" * 100)
    logger.info("🌍 STARTING MULTI-DATASET PRF EVALUATION")
    logger.info("=" * 100)
    logger.info(f"📊 Datasets: {', '.join(datasets)}")
    logger.info(f"📈 Total datasets to process: {len(datasets)}")
    logger.info("=" * 100 + "\n")
    
    overall_start_time = time.time()
    results_summary = []
    
    for dataset_idx, dataset_name in enumerate(datasets, 1):
        try:
            logger.info(f"\n{'🔥' * 50}")
            logger.info(f"🎯 DATASET {dataset_idx}/{len(datasets)}: {dataset_name}")
            logger.info(f"{'🔥' * 50}")
            
            dataset_start_time = time.time()
            
            # Initialize evaluator for this dataset
            evaluator = PerformancePRFEvaluator(dataset_name)
            
            # Run evaluation for all configurations
            evaluator.evaluate_all_configurations(k=k)
            
            dataset_elapsed_time = time.time() - dataset_start_time
            
            # Record results
            dataset_result = {
                'dataset': dataset_name,
                'status': 'completed',
                'elapsed_time_seconds': dataset_elapsed_time,
                'elapsed_time_formatted': f"{dataset_elapsed_time/60:.1f} minutes",
                'num_images': len(evaluator.selected_images),
                'num_llm_models': len(evaluator.llm_models),
                'num_rocchio_configs': len(evaluator.rocchio_params),
                'total_evaluations': len(evaluator.llm_models) * len(evaluator.rocchio_params)
            }
            results_summary.append(dataset_result)
            
            logger.info(f"✅ {dataset_name} evaluation completed in {dataset_elapsed_time/60:.1f} minutes")
            
        except Exception as e:
            dataset_elapsed_time = time.time() - dataset_start_time
            logger.error(f"❌ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Record error
            dataset_result = {
                'dataset': dataset_name,
                'status': 'failed',
                'error': str(e),
                'elapsed_time_seconds': dataset_elapsed_time,
                'elapsed_time_formatted': f"{dataset_elapsed_time/60:.1f} minutes"
            }
            results_summary.append(dataset_result)
            
            logger.warning(f"⚠️ Continuing with next dataset after {dataset_name} failure...")
    
    # Calculate overall statistics
    overall_elapsed_time = time.time() - overall_start_time
    completed_datasets = [r for r in results_summary if r['status'] == 'completed']
    failed_datasets = [r for r in results_summary if r['status'] == 'failed']
    
    # Save overall summary
    overall_summary = {
        'evaluation_type': 'multi_dataset_prf',
        'evaluation_date': datetime.now().isoformat(),
        'total_datasets': len(datasets),
        'completed_datasets': len(completed_datasets),
        'failed_datasets': len(failed_datasets),
        'overall_elapsed_time_seconds': overall_elapsed_time,
        'overall_elapsed_time_formatted': f"{overall_elapsed_time/3600:.2f} hours",
        'k_value': k,
        'dataset_results': results_summary
    }
    
    # Save to report directory
    summary_dir = Config.REPORT_DIR / "performance_prf"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / f"multi_dataset_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    
    # Display final results
    logger.info("\n" + "=" * 100)
    logger.info("🎉 MULTI-DATASET EVALUATION COMPLETE")
    logger.info("=" * 100)
    logger.info(f"⏱️  Total time: {overall_elapsed_time/3600:.2f} hours")
    logger.info(f"✅ Completed: {len(completed_datasets)}/{len(datasets)} datasets")
    
    if completed_datasets:
        logger.info(f"\n📊 COMPLETED DATASETS:")
        for result in completed_datasets:
            logger.info(f"  ✅ {result['dataset']}: {result['elapsed_time_formatted']} "
                       f"({result['total_evaluations']} evaluations, {result['num_images']} images)")
    
    if failed_datasets:
        logger.info(f"\n❌ FAILED DATASETS:")
        for result in failed_datasets:
            logger.info(f"  ❌ {result['dataset']}: {result['error']}")
    
    logger.info(f"\n📁 Summary saved to: {summary_file}")
    logger.info("=" * 100)
    
    return overall_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PRF retrieval performance across different LLMs and Rocchio parameters"
    )
    parser.add_argument(
        '--dataset',
        choices=['COCO', 'Flickr', 'VizWiz'],
        help='Dataset to evaluate (COCO, Flickr, or VizWiz). Not required when using --all-datasets'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to retrieve (default: 10)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default="gemma3:4b",
        help='Evaluate a single LLM model (default: all models)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Rocchio alpha parameter (requires --beta)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='Rocchio beta parameter (requires --alpha)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.0,
        help='Rocchio gamma parameter (default: 0.0)'
    )
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Evaluate all datasets (COCO, Flickr, VizWiz) sequentially'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all_datasets and not args.dataset:
        parser.error("Either --dataset or --all-datasets must be specified")

    try:
        # Check if all datasets evaluation requested
        if args.all_datasets:
            # Evaluate all datasets sequentially
            logger.info("🌍 Starting evaluation on all datasets...")
            evaluate_all_datasets(k=args.k)
            return 0
        
        # Single dataset evaluation (original behavior)
        evaluator = PerformancePRFEvaluator(args.dataset)

        # Check if specific configuration requested
        if args.llm_model and args.alpha is not None and args.beta is not None:
            # Evaluate single configuration
            logger.info(f"🎯 Evaluating single configuration:")
            logger.info(f"  LLM: {args.llm_model}")
            logger.info(f"  Rocchio: α={args.alpha}, β={args.beta}, γ={args.gamma}")

            # Override evaluator settings for single run
            evaluator.llm_models = [args.llm_model]
            evaluator.rocchio_params = [(args.alpha, args.beta, args.gamma)]

        # Run evaluation
        evaluator.evaluate_all_configurations(k=args.k)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
