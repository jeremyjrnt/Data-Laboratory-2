#!/usr/bin/env python3
"""
Performance Evaluation for Combined Structured + Unstructured Retriever
Evaluates the two-stage retrieval system on image retrieval.

python src\performance\performance_structured_unstructured.py --dataset COCO

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

from retrieval.retriever_structured_unstructured import CombinedImageRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceStructuredUnstructuredEvaluator:
    """
    Evaluates combined structured + unstructured retrieval performance.
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
            "mistral:7b",
            "gemma3:4b",
            "gemma3:27b"
        ]

        # Setup directories
        self.output_dir = Path(f"report/performance_structured_unstructured/{dataset_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Single consolidated results file for this dataset
        self.results_file = self.output_dir / f"performance_structured_unstructured_{dataset_name}.json"

        # Load selected test cases
        self.selected_images = self._load_selected_images()
        
        # Load existing results if file exists, otherwise initialize empty
        self.all_results = self._load_existing_results()

        logger.info(f"📊 Performance Structured+Unstructured Evaluator initialized for {dataset_name}")
        logger.info(f"🖼️  Test images: {len(self.selected_images)}")
        logger.info(f"🤖 LLM models: {self.llm_models}")
        if self.all_results:
            existing_models = [r.get("llm_model") for r in self.all_results]
            logger.info(f"📥 Loaded existing results for: {existing_models}")

    def _load_selected_images(self) -> List[Dict]:
        """Load the selected_1000.json file for the dataset."""
        selected_path = Path(f"data/{self.dataset_name}/selected_1000.json")

        if not selected_path.exists():
            raise FileNotFoundError(f"Selected images file not found: {selected_path}")

        with open(selected_path, 'r', encoding='utf-8') as f:
            images = json.load(f)

        logger.info(f"📂 Loaded {len(images)} images from {selected_path}")
        return images

    def _load_existing_results(self) -> List[Dict]:
        """Load existing results from JSON file if it exists."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                existing_results = data.get("evaluations", [])
                logger.info(f"📥 Loaded {len(existing_results)} existing evaluation(s) from {self.results_file}")
                return existing_results
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing results: {e}")
                return []
        return []

    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use in filenames (Windows-compatible)."""
        invalid_chars = [':', '*', '?', '"', '<', '>', '|', '\\', '/']
        sanitized = model_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized

    def evaluate_single_configuration(
        self,
        llm_model: str,
        llm_temp: float = 0.1,
        llm_top_p: float = 0.9,
        llm_timeout: int = 200
    ) -> Dict:
        """
        Evaluate retrieval performance for a single LLM configuration.

        Args:
            llm_model: LLM model name
            llm_temp: LLM temperature
            llm_top_p: LLM top_p parameter
            llm_timeout: LLM timeout in seconds

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("\n" + "="*80)
        logger.info(f"🤖 Evaluating LLM: {llm_model}")
        logger.info("="*80)

        # Initialize retriever
        retriever = CombinedImageRetriever(
            vectordb_name=self.vectordb_name,
            vectordb_dir="VectorDBs",
            llm_model=llm_model,
            llm_temp=llm_temp,
            llm_top_p=llm_top_p,
            llm_timeout=llm_timeout
        )

        # Check if we have existing results for this model
        existing_model_eval = None
        for eval_entry in self.all_results:
            if eval_entry.get("llm_model") == llm_model:
                existing_model_eval = eval_entry
                break
        
        # Load existing results or start fresh
        if existing_model_eval:
            results = existing_model_eval.get("results", [])
            already_processed = {r.get("filename") for r in results if "filename" in r}
            logger.info(f"📥 Found {len(results)} existing results for {llm_model}")
            logger.info(f"▶️  Resuming from image {len(results) + 1}")
        else:
            results = []
            already_processed = set()
        
        total_images = len(self.selected_images)

        for idx, image_data in enumerate(self.selected_images, 1):
            filename = image_data.get("filename")
            caption = image_data.get("caption")
            baseline_rank = image_data.get("baseline_rank")

            if not filename or not caption:
                logger.warning(f"⚠️ Skipping image {idx}: missing filename or caption")
                continue
            
            # Skip if already processed
            if filename in already_processed:
                logger.info(f"⏭️  Skipping image {idx}/{total_images}: {filename} (already processed)")
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"📸 Image {idx}/{total_images}: {filename}")
            logger.info(f"📝 Query: {caption}")
            logger.info(f"🎯 Baseline rank: {baseline_rank}")
            logger.info(f"{'='*80}")

            try:
                # Run combined two-stage retrieval
                start_time = time.time()
                result = retriever.combined_two_stage_retrieve(
                    query_text=caption,
                    target_filename=filename
                )
                elapsed_time = time.time() - start_time

                # Add metadata
                result["image_index"] = idx
                result["filename"] = filename
                result["query"] = caption
                result["baseline_rank"] = baseline_rank
                result["elapsed_time"] = elapsed_time

                results.append(result)

                # Save after each image
                self._save_single_image_result(llm_model, result)

                # Log summary
                logger.info(f"\n{'='*80}")
                logger.info(f"✅ Completed {filename} in {elapsed_time:.2f}s")
                logger.info(f"{'='*80}\n")

            except Exception as e:
                logger.error(f"❌ Error processing {filename}: {e}", exc_info=True)
                error_result = {
                    "image_index": idx,
                    "filename": filename,
                    "query": caption,
                    "baseline_rank": baseline_rank,
                    "error": str(e)
                }
                results.append(error_result)
                
                # Save error result too
                self._save_single_image_result(llm_model, error_result)

        # Compile evaluation summary
        evaluation_result = {
            "llm_model": llm_model,
            "llm_temp": llm_temp,
            "llm_top_p": llm_top_p,
            "llm_timeout": llm_timeout,
            "dataset": self.dataset_name,
            "total_images": total_images,
            "successful_evaluations": sum(1 for r in results if "error" not in r),
            "failed_evaluations": sum(1 for r in results if "error" in r),
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

        return evaluation_result

    def run_all_evaluations(self):
        """Run evaluations for all LLM models and save consolidated results."""
        logger.info("\n" + "="*80)
        logger.info("🚀 Starting Performance Evaluation")
        logger.info(f"📊 Dataset: {self.dataset_name}")
        logger.info(f"🖼️  Test Images: {len(self.selected_images)}")
        logger.info(f"🤖 LLM Models: {len(self.llm_models)}")
        logger.info("="*80)

        # Get list of already completed models (with all images processed)
        completed_models = set()
        for r in self.all_results:
            model_name = r.get("llm_model")
            num_results = len(r.get("results", []))
            if num_results >= len(self.selected_images):
                completed_models.add(model_name)
        
        for llm_model in self.llm_models:
            # Skip if already completed (all images processed)
            if llm_model in completed_models:
                logger.info(f"\n⏭️  Skipping {llm_model} (already completed with all {len(self.selected_images)} images)")
                continue
            
            # Check if partially completed
            partial_eval = None
            for r in self.all_results:
                if r.get("llm_model") == llm_model:
                    partial_eval = r
                    break
            
            if partial_eval:
                num_existing = len(partial_eval.get("results", []))
                logger.info(f"\n▶️  Resuming {llm_model} (found {num_existing}/{len(self.selected_images)} images)")
                
            try:
                result = self.evaluate_single_configuration(llm_model)
                
                # Update or add the result
                updated = False
                for i, existing_result in enumerate(self.all_results):
                    if existing_result.get("llm_model") == llm_model:
                        self.all_results[i] = result
                        updated = True
                        break
                
                if not updated:
                    self.all_results.append(result)

            except Exception as e:
                logger.error(f"❌ Failed to evaluate LLM {llm_model}: {e}", exc_info=True)

        logger.info("\n" + "="*80)
        logger.info("✅ ALL EVALUATIONS COMPLETE")
        logger.info(f"📁 Results saved to: {self.results_file}")
        logger.info("="*80)

        return self.all_results
    
    def _save_single_image_result(self, llm_model: str, image_result: Dict):
        """Save results after each image evaluation (incremental save)."""
        # Find or create the evaluation entry for this LLM model
        model_eval = None
        for eval_entry in self.all_results:
            if eval_entry.get("llm_model") == llm_model:
                model_eval = eval_entry
                break
        
        # If this is the first image for this model, create the entry
        if model_eval is None:
            model_eval = {
                "llm_model": llm_model,
                "llm_temp": 0.1,
                "llm_top_p": 0.9,
                "llm_timeout": 200,
                "dataset": self.dataset_name,
                "total_images": len(self.selected_images),
                "results": []
            }
            self.all_results.append(model_eval)
        
        # Add the new image result
        model_eval["results"].append(image_result)
        
        # Update summary stats
        model_eval["successful_evaluations"] = sum(1 for r in model_eval["results"] if "error" not in r)
        model_eval["failed_evaluations"] = sum(1 for r in model_eval["results"] if "error" in r)
        model_eval["timestamp"] = datetime.now().isoformat()
        
        # Save the entire consolidated file
        self._save_results(self.all_results)

    def _save_results(self, results: List[Dict]):
        """Save results to JSON file."""
        consolidated = {
            "dataset": self.dataset_name,
            "total_llm_models": len(self.llm_models),
            "completed_evaluations": len(results),
            "timestamp": datetime.now().isoformat(),
            "evaluations": results
        }

        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)

        # Silent save - no log message to avoid spam


def main():
    parser = argparse.ArgumentParser(
        description="Performance Evaluation for Structured + Unstructured Retriever"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["COCO", "Flickr", "VizWiz"],
        help="Dataset to evaluate (COCO, Flickr, or VizWiz)"
    )

    args = parser.parse_args()

    # Create evaluator and run
    evaluator = PerformanceStructuredUnstructuredEvaluator(args.dataset)
    evaluator.run_all_evaluations()


if __name__ == "__main__":
    main()
