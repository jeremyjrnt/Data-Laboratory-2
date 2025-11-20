#!/usr/bin/env python3
"""
Performance Evaluation for LLM Voting Retriever
Evaluates different LLM models on image-caption pairs where ground truth is ranked 2-5 by CLIP similarity.
Optimized version with BLIP caching and progressive saving.
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
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retriever_vote import GroundTruthLLMVotingRetriever
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedPerformanceVoteEvaluator:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.vectordb_name = f"{dataset_name}_VectorDB"
        
        # LLM models to test
        self.llm_models = [
            "mistral:7b",
            "gpt-oss:20b",
            "gemma3:4b",
            "gemma3:27b"
        ]
        
        # Target ranks for evaluation (ground truth not in position 1)
        self.target_ranks = [2, 3, 4, 5]
        self.cases_per_rank = Config.EVALUATION_CASES_PER_RANK  # From Config
        
        # Setup directories
        self.output_dir = Config.REPORT_DIR / "performance_vote" / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Config.REPORT_DIR / "performance_vote" / dataset_name / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and select test cases
        self.performance_data = self._load_performance_data()
        self.selected_test_cases = self._select_test_cases()
        self.blip_cache_file = self.temp_dir / "blip_descriptions_cache.pkl"
        
        logger.info(f"📊 Selected {len(self.selected_test_cases)} test cases for {dataset_name}")
        logger.info(f"🎯 Cases per rank: {self.cases_per_rank}")
        logger.info(f"🤖 LLM models: {self.llm_models}")
    
    def _load_performance_data(self) -> List[Dict]:
        """Load performance.json data for the dataset."""
        performance_path = Config.REPORT_DIR / "performance_raw" / self.dataset_name / "performance.json"
        
        if not performance_path.exists():
            raise FileNotFoundError(f"Performance data not found: {performance_path}")
        
        with open(performance_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract image_results array
        if isinstance(data, dict) and 'image_results' in data:
            image_results = data['image_results']
        else:
            raise ValueError("Expected 'image_results' key in performance data")
        
        logger.info(f"📁 Loaded performance data: {len(image_results)} entries")
        return image_results
    
    def _select_test_cases(self) -> List[Dict]:
        """Select exactly 1000 test cases per rank (2-5) for comprehensive evaluation."""
        selected_cases = []
        
        for rank in self.target_ranks:
            # Get all cases for this rank
            rank_cases = [entry for entry in self.performance_data 
                         if entry.get('exact_rank') == rank]
            
            # Select first 1000 cases (or all if less than 1000)
            selected_rank_cases = rank_cases[:self.cases_per_rank]
            
            logger.info(f"📋 Rank {rank}: {len(selected_rank_cases)} cases selected (available: {len(rank_cases)})")
            
            if len(selected_rank_cases) < self.cases_per_rank:
                logger.warning(f"⚠️ Only {len(selected_rank_cases)} cases available for rank {rank}, requested {self.cases_per_rank}")
            
            for case in selected_rank_cases:
                selected_cases.append({
                    'image_id': case['image_id'],
                    'filename': case['filename'], 
                    'caption': case['caption'],
                    'exact_rank': case['exact_rank'],
                    'similarity': case.get('similarity', 0.0)
                })
        
        # Sort by exact_rank for organized evaluation
        selected_cases.sort(key=lambda x: (x['exact_rank'], x['image_id']))
        
        return selected_cases
    
    def _get_k_for_rank(self, exact_rank: int) -> int:
        """Get the number of results to retrieve based on exact rank."""
        k_mapping = {
            2: 3,  # Rank 2 -> top 3 retrieved
            3: 5,  # Rank 3 -> top 5 retrieved  
            4: 7,  # Rank 4 -> top 7 retrieved
            5: 9   # Rank 5 -> top 9 retrieved
        }
        return k_mapping.get(exact_rank, 5)  # Default to 5 if not found
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use in filenames (Windows-compatible)."""
        # Replace characters that are invalid in Windows filenames
        invalid_chars = [':', '*', '?', '"', '<', '>', '|', '\\', '/']
        sanitized = model_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized
    
    def _generate_and_cache_blip_descriptions(self):
        """Generate BLIP descriptions once and cache them."""
        
        if self.blip_cache_file.exists():
            logger.info(f"📁 Loading cached BLIP descriptions from {self.blip_cache_file}")
            with open(self.blip_cache_file, 'rb') as f:
                blip_cache = pickle.load(f)
            
            # Check if cache is complete
            cached_files = set(blip_cache.keys())
            required_files = set(case['filename'] for case in self.selected_test_cases)
            
            if required_files.issubset(cached_files):
                logger.info(f"✅ BLIP cache is complete ({len(blip_cache)} descriptions)")
                return blip_cache
            else:
                missing = required_files - cached_files
                logger.info(f"⚠️ BLIP cache incomplete, missing {len(missing)} descriptions")
        else:
            logger.info(f"� Creating new BLIP descriptions cache")
            blip_cache = {}
        
        # Initialize retriever for BLIP generation only
        logger.info("🔄 Initializing retriever for BLIP generation...")
        temp_retriever = GroundTruthLLMVotingRetriever(
            vectordb_name=self.vectordb_name,
            llm_model="gemma3:27b"  # Use any model, we only need BLIP
        )
        
        # Generate missing descriptions
        total_cases = len(self.selected_test_cases)
        for i, test_case in enumerate(self.selected_test_cases, 1):
            filename = test_case['filename']
            # Get k based on exact rank
            k = self._get_k_for_rank(test_case['exact_rank'])
            
            if filename in blip_cache:
                # Check if cached data has the correct number of images for this rank
                cached_data = blip_cache[filename]
                if len(cached_data.get('similar_images', [])) != k:
                    logger.info(f"🔄 Regenerating cache for {filename}: need k={k}, cached has {len(cached_data.get('similar_images', []))}")
                else:
                    continue  # Skip if already cached with correct k
            else:
                logger.info(f"🖼️ [{i}/{total_cases}] Generating BLIP descriptions for {filename} (rank {test_case['exact_rank']}, k={k})")
            
            try:
                # Get ground truth info
                result = temp_retriever.get_caption_by_filename(filename)
                if not result:
                    logger.warning(f"⚠️ Caption not found for {filename}")
                    continue
                
                caption, ground_truth_id, metadata = result
                
                # Get similar images (this will include BLIP generation)
                similar_images = temp_retriever.retrieve_similar_images(caption, k=k)
                
                # Check if ground truth found
                ground_truth_found = any(
                    img['faiss_index'] == ground_truth_id for img in similar_images
                )
                
                if not ground_truth_found:
                    logger.warning(f"⚠️ Ground truth not in top-{k} for {filename}")
                    continue
                
                # Generate BLIP descriptions
                for img_info in similar_images:
                    image_path = temp_retriever.images_dir / img_info['filename']
                    if image_path.exists():
                        blip_caption = temp_retriever.generate_blip_caption(image_path)
                        img_info['blip_description'] = blip_caption
                        img_info['image_path'] = str(image_path)
                    else:
                        img_info['blip_description'] = "Image file not found"
                        img_info['image_path'] = None
                
                # Cache the results
                blip_cache[filename] = {
                    'caption': caption,
                    'ground_truth_id': ground_truth_id,
                    'similar_images': similar_images,
                    'k_used': k,
                    'exact_rank': test_case['exact_rank'],
                    'generated_at': datetime.now().isoformat()
                }
                
                # Save cache every 50 items (more frequent for 1000 cases)
                if i % 50 == 0:
                    with open(self.blip_cache_file, 'wb') as f:
                        pickle.dump(blip_cache, f)
                    logger.info(f"💾 Cached {len(blip_cache)} BLIP descriptions")
                
            except Exception as e:
                logger.error(f"❌ Error processing {filename}: {e}")
                continue
        
        # Final save
        with open(self.blip_cache_file, 'wb') as f:
            pickle.dump(blip_cache, f)
        
        logger.info(f"✅ BLIP descriptions cached: {len(blip_cache)} files")
        return blip_cache
    
    def _save_progress(self, llm_model: str, results: List[Dict], consolidated_results: Dict = None):
        """Save progress incrementally in consolidated format."""
        # Update consolidated results immediately if provided
        if consolidated_results is not None:
            # Update the current model's results in consolidated structure
            consolidated_results['llm_evaluations'][llm_model] = {
                'llm_model': llm_model,
                'status': 'in_progress',
                'completed_cases': len(results),
                'timestamp': datetime.now().isoformat(),
                'results': results  # Save all results so far
            }
            self._update_consolidated_results(consolidated_results)
            logger.info(f"💾 Progress saved: {len(results)} results for {llm_model} → consolidated_results.json")
    
    def _update_consolidated_results(self, consolidated_results: Dict):
        """Update the consolidated results file."""
        consolidated_file = self.output_dir / "consolidated_results.json"
        
        # Add metadata if not present
        if 'metadata' not in consolidated_results:
            consolidated_results['metadata'] = {
                'dataset': self.dataset_name,
                'start_time': datetime.now().isoformat(),
                'target_ranks': self.target_ranks,
                'cases_per_rank': self.cases_per_rank,
                'total_models': len(self.llm_models),
                'models': self.llm_models
            }
        
        consolidated_results['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save consolidated results (main file)
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Consolidated results updated: {consolidated_file}")
        logger.info(f"� Progress automatically saved to single JSON file")

    def _load_consolidated_results(self) -> Dict:
        """Load existing consolidated results or create new structure."""
        consolidated_file = self.output_dir / "consolidated_results.json"
        
        if consolidated_file.exists():
            try:
                with open(consolidated_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing consolidated results: {e}")
        
        # Create new consolidated structure
        return {
            'metadata': {
                'dataset': self.dataset_name,
                'start_time': datetime.now().isoformat(),
                'target_ranks': self.target_ranks,
                'cases_per_rank': self.cases_per_rank,
                'total_models': len(self.llm_models),
                'models': self.llm_models
            },
            'llm_evaluations': {},
            'summary': {}
        }
    
    def evaluate_llm_model(self, llm_model: str, blip_cache: Dict, consolidated_results: Dict = None) -> Dict:
        """Evaluate a specific LLM model using cached BLIP descriptions."""
        logger.info(f"\n🚀 Evaluating LLM: {llm_model}")
        logger.info("=" * 60)
        
        try:
            # Initialize retriever with specific LLM
            retriever = GroundTruthLLMVotingRetriever(
                vectordb_name=self.vectordb_name,
                llm_model=llm_model
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize retriever for {llm_model}: {e}")
            return {
                'llm_model': llm_model,
                'status': 'initialization_failed',
                'error': str(e),
                'results': []
            }
        
        results = []
        total_cases = len(self.selected_test_cases)
        successful_cases = 0
        improvements = 0
        degradations = 0
        
        logger.info(f"📊 Total test cases for {llm_model}: {total_cases}")
        logger.info(f"🎯 Expected total cases: {total_cases} (1000 per rank × 4 ranks)")
        
        start_time = time.time()
        
        for i, test_case in enumerate(self.selected_test_cases, 1):
            filename = test_case['filename']
            
            if filename not in blip_cache:
                logger.warning(f"⚠️ [{i}/{total_cases}] No BLIP cache for {filename}")
                continue
            
            logger.info(f"🔄 [{i}/{total_cases}] Testing {filename} (rank {test_case['exact_rank']})")
            
            try:
                # Get cached data
                cached_data = blip_cache[filename]
                caption = cached_data['caption']
                ground_truth_id = cached_data['ground_truth_id']
                similar_images_all = cached_data['similar_images']
                
                # Get adaptive k for this test case
                k = self._get_k_for_rank(test_case['exact_rank'])
                
                # Use only the first k images according to adaptive strategy
                similar_images = similar_images_all[:k]
                
                logger.info(f"   Using k={k} images for rank {test_case['exact_rank']} (adaptive)")
                
                # Find ground truth position in CLIP results
                ground_truth_clip_rank = None
                for img in similar_images:
                    if img['faiss_index'] == ground_truth_id:
                        ground_truth_clip_rank = img['clip_rank']
                        break
                
                if ground_truth_clip_rank is None:
                    logger.warning(f"   ❌ Ground truth not found in cached results")
                    continue
                
                # If already rank 1, skip LLM voting
                if ground_truth_clip_rank == 1:
                    result_entry = {
                        'image_id': test_case['image_id'],
                        'filename': filename,
                        'caption': caption,
                        'expected_rank': test_case['exact_rank'],
                        'clip_rank': 1,
                        'llm_rank': 1,
                        'improvement': 0,
                        'clip_similarity': similar_images[0]['similarity'],
                        'llm_decision_time_seconds': 0.0,
                        'status': 'already_rank_1'
                    }
                    results.append(result_entry)
                    logger.info(f"   ✅ Already rank #1, no improvement needed")
                    continue
                
                # Create voting prompt with dynamic number of images and call LLM with retry
                logger.info(f"   🤖 Asking {llm_model} to rank...")
                llm_start_time = time.time()
                ranking, llm_response = retriever.get_llm_ranking_with_retry(caption, similar_images)
                llm_decision_time = time.time() - llm_start_time
                
                # Find new ground truth position
                ground_truth_llm_rank = None
                
                # Check if we got a fallback ranking (CLIP order preserved)
                is_fallback = (ranking == list(range(1, len(similar_images) + 1)) and 
                              "FALLBACK" in llm_response)
                
                if is_fallback:
                    # For fallback, preserve CLIP ranking (no improvement)
                    ground_truth_llm_rank = ground_truth_clip_rank
                    logger.info(f"   🔄 Using fallback: LLM ranking = CLIP ranking = {ground_truth_llm_rank}")
                else:
                    # Normal case: find ground truth in LLM ranking
                    for llm_rank, temp_idx in enumerate(ranking, 1):
                        # Find the result with this temp_index
                        for img in similar_images:
                            if img['temp_index'] == temp_idx and img['faiss_index'] == ground_truth_id:
                                ground_truth_llm_rank = llm_rank
                                break
                        if ground_truth_llm_rank:
                            break
                
                if ground_truth_llm_rank is None:
                    logger.warning(f"   ❌ Could not find ground truth in LLM ranking, using fallback")
                    # Fallback: preserve CLIP ranking
                    ground_truth_llm_rank = ground_truth_clip_rank
                    ranking = list(range(1, len(similar_images) + 1))
                    llm_response = f"FALLBACK: Could not map ground truth in ranking"
                    is_fallback = True
                
                improvement = ground_truth_clip_rank - ground_truth_llm_rank
                
                # Determine status based on whether fallback was used
                status = 'fallback_used' if is_fallback else 'success'
                
                result_entry = {
                    'image_id': test_case['image_id'],
                    'filename': filename,
                    'caption': caption,
                    'expected_rank': test_case['exact_rank'],
                    'clip_rank': ground_truth_clip_rank,
                    'llm_rank': ground_truth_llm_rank,
                    'improvement': improvement,
                    'clip_similarity': next(img['similarity'] for img in similar_images 
                                          if img['faiss_index'] == ground_truth_id),
                    'llm_response': llm_response,
                    'llm_ranking': ranking,
                    'llm_decision_time_seconds': round(llm_decision_time, 3),
                    'status': status
                }
                
                results.append(result_entry)
                successful_cases += 1
                
                if is_fallback:
                    logger.info(f"   🔄 Fallback used: {ground_truth_clip_rank} → {ground_truth_llm_rank} (no change)")
                elif improvement > 0:
                    improvements += 1
                    logger.info(f"   ✅ Improved: {ground_truth_clip_rank} → {ground_truth_llm_rank} (+{improvement})")
                elif improvement < 0:
                    degradations += 1
                    logger.info(f"   ⚠️ Degraded: {ground_truth_clip_rank} → {ground_truth_llm_rank} ({improvement})")
                else:
                    logger.info(f"   ➖ No change: {ground_truth_clip_rank} → {ground_truth_llm_rank}")
                
                # Save progress after EVERY case to consolidated file
                self._save_progress(llm_model, results, consolidated_results)
                
                # Log intermediate statistics every 100 cases for better progress tracking
                if successful_cases > 0 and successful_cases % 100 == 0:
                    current_improvement_rate = improvements / successful_cases
                    elapsed_so_far = time.time() - start_time
                    avg_time_per_case = elapsed_so_far / i if i > 0 else 0
                    remaining_cases = total_cases - i
                    estimated_remaining_time = avg_time_per_case * remaining_cases
                    
                    logger.info(f"   📊 Progress milestone: {successful_cases} successful cases completed ({i}/{total_cases} total)")
                    logger.info(f"   📈 Current improvement rate: {current_improvement_rate:.1%} ({improvements} improvements)")
                    logger.info(f"   📉 Current degradation rate: {degradations/successful_cases:.1%} ({degradations} degradations)")
                    logger.info(f"   ⏱️ Elapsed: {elapsed_so_far:.1f}s | Est. remaining: {estimated_remaining_time:.1f}s")
            
            except Exception as e:
                logger.error(f"   ❌ Error processing {filename}: {e}")
                results.append({
                    'image_id': test_case['image_id'],
                    'filename': filename,
                    'expected_rank': test_case['exact_rank'],
                    'llm_decision_time_seconds': 0.0,
                    'status': 'error',
                    'error': str(e)
                })
                
                # Save progress even for errors
                self._save_progress(llm_model, results, consolidated_results)
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        improvement_rate = improvements / successful_cases if successful_cases > 0 else 0
        degradation_rate = degradations / successful_cases if successful_cases > 0 else 0
        
        # Calculate LLM timing statistics
        llm_decision_times = [r['llm_decision_time_seconds'] for r in results 
                             if 'llm_decision_time_seconds' in r and r['llm_decision_time_seconds'] > 0]
        
        timing_stats = {}
        if llm_decision_times:
            timing_stats = {
                'total_llm_calls': len(llm_decision_times),
                'min_decision_time': round(min(llm_decision_times), 3),
                'max_decision_time': round(max(llm_decision_times), 3),
                'avg_decision_time': round(sum(llm_decision_times) / len(llm_decision_times), 3),
                'total_llm_time': round(sum(llm_decision_times), 3)
            }
        
        summary = {
            'llm_model': llm_model,
            'status': 'completed',
            'total_test_cases': total_cases,
            'successful_evaluations': successful_cases,
            'improvements': improvements,
            'degradations': degradations,
            'no_changes': successful_cases - improvements - degradations,
            'improvement_rate': improvement_rate,
            'degradation_rate': degradation_rate,
            'elapsed_time_seconds': elapsed_time,
            'llm_timing_stats': timing_stats,
            'results': results
        }
        
        # Final progress save with completed status
        summary['status'] = 'completed'
        summary['timestamp'] = datetime.now().isoformat()
        
        if consolidated_results is not None:
            # Update with final completed status
            consolidated_results['llm_evaluations'][llm_model] = summary
            self._update_consolidated_results(consolidated_results)
        
        logger.info(f"\n📊 Summary for {llm_model}:")
        logger.info(f"   ✅ Successful evaluations: {successful_cases}/{total_cases}")
        logger.info(f"   📈 Improvements: {improvements} ({improvement_rate:.1%})")
        logger.info(f"   📉 Degradations: {degradations} ({degradation_rate:.1%})")
        logger.info(f"   ⏱️ Time: {elapsed_time:.1f}s")
        
        if timing_stats:
            logger.info(f"   🤖 LLM timing: {timing_stats['total_llm_calls']} calls")
            logger.info(f"      ⏱️ Avg decision time: {timing_stats['avg_decision_time']}s")
            logger.info(f"      ⏱️ Min/Max: {timing_stats['min_decision_time']}s / {timing_stats['max_decision_time']}s")
            logger.info(f"      ⏱️ Total LLM time: {timing_stats['total_llm_time']}s")
        
        return summary
    
    def run_full_evaluation(self) -> Dict:
        """Run evaluation on all LLM models with consolidated results."""
        logger.info(f"\n🎯 Starting Optimized LLM Voting Performance Evaluation")
        logger.info(f"📊 Dataset: {self.dataset_name}")
        logger.info(f"🔢 Cases per rank: {self.cases_per_rank}")
        logger.info("=" * 80)
        
        # Load or create consolidated results structure
        consolidated_results = self._load_consolidated_results()
        
        # Generate BLIP descriptions once
        logger.info("\n📝 Phase 1: Generating BLIP descriptions...")
        blip_cache = self._generate_and_cache_blip_descriptions()
        
        # Update metadata with BLIP info
        consolidated_results['metadata'].update({
            'blip_cache_size': len(blip_cache),
            'total_selected_cases': len(self.selected_test_cases)
        })
        
        overall_start = time.time()
        
        # Phase 2: Evaluate each LLM
        logger.info(f"\n🤖 Phase 2: Evaluating LLM models...")
        
        for llm_model in self.llm_models:
            try:
                logger.info(f"\n{'='*20} {llm_model} {'='*20}")
                model_results = self.evaluate_llm_model(llm_model, blip_cache, consolidated_results)
                
                # Update consolidated results
                consolidated_results['llm_evaluations'][llm_model] = model_results
                consolidated_results['metadata']['last_updated'] = datetime.now().isoformat()
                
                # Save consolidated results after each model
                self._update_consolidated_results(consolidated_results)
                
            except KeyboardInterrupt:
                logger.warning(f"⚠️ Evaluation interrupted for {llm_model}")
                consolidated_results['llm_evaluations'][llm_model] = {
                    'llm_model': llm_model,
                    'status': 'interrupted',
                    'timestamp': datetime.now().isoformat()
                }
                self._update_consolidated_results(consolidated_results)
                break
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {llm_model}: {e}")
                consolidated_results['llm_evaluations'][llm_model] = {
                    'llm_model': llm_model,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self._update_consolidated_results(consolidated_results)
        
        total_elapsed = time.time() - overall_start
        consolidated_results['metadata']['total_elapsed_time_seconds'] = total_elapsed
        consolidated_results['metadata']['completion_time'] = datetime.now().isoformat()
        
        # Generate summary statistics
        completed_models = [k for k, v in consolidated_results['llm_evaluations'].items() 
                          if v.get('status') == 'completed']
        
        if completed_models:
            total_improvements = sum(v.get('improvements', 0) for v in consolidated_results['llm_evaluations'].values())
            total_evaluations = sum(v.get('successful_evaluations', 0) for v in consolidated_results['llm_evaluations'].values())
            
            consolidated_results['summary'] = {
                'completed_models': len(completed_models),
                'total_models': len(self.llm_models),
                'total_improvements': total_improvements,
                'total_evaluations': total_evaluations,
                'overall_improvement_rate': total_improvements / total_evaluations if total_evaluations > 0 else 0,
                'completion_status': 'completed' if len(completed_models) == len(self.llm_models) else 'partial'
            }
        
        # Final save
        self._update_consolidated_results(consolidated_results)
        
        logger.info(f"\n🏁 Full evaluation completed in {total_elapsed:.1f}s")
        logger.info(f"📊 Completed models: {len(completed_models)}/{len(self.llm_models)}")
        
        return consolidated_results
    
    def save_results(self, consolidated_results: Dict, output_dir: Path = None):
        """Save evaluation results to JSON file."""
        if output_dir is None:
            output_dir = self.output_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"llm_voting_performance_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Final results archived: {output_file}")
        logger.info(f"📊 Main consolidated file: {self.output_dir / 'consolidated_results.json'}")
        logger.info(f"💾 All progress saved in single JSON file during execution")
        
        # Print summary
        self._print_summary(consolidated_results)
        
        return output_file
    
    def _print_summary(self, consolidated_results: Dict):
        """Print a nice summary of the results."""
        print("\n" + "=" * 80)
        print("🏆 LLM VOTING PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"📊 Dataset: {consolidated_results.get('metadata', {}).get('dataset', 'Unknown')}")
        print(f"📅 Start time: {consolidated_results.get('metadata', {}).get('start_time', 'Unknown')}")
        print(f"⏱️ Total time: {consolidated_results.get('metadata', {}).get('total_elapsed_time_seconds', 0):.1f}s")
        print(f"🔢 Cases per rank: {consolidated_results.get('metadata', {}).get('cases_per_rank', 0)}")
        print(f"📁 BLIP cache size: {consolidated_results.get('metadata', {}).get('blip_cache_size', 0)}")
        
        print(f"\n📈 RESULTS BY LLM MODEL:")
        print("-" * 80)
        
        for llm_model, results in consolidated_results.get('llm_evaluations', {}).items():
            if results['status'] == 'completed':
                print(f"\n🤖 {llm_model}:")
                print(f"   ✅ Success rate: {results['successful_evaluations']}/{results['total_test_cases']}")
                print(f"   📈 Improvements: {results['improvements']} ({results['improvement_rate']:.1%})")
                print(f"   📉 Degradations: {results['degradations']} ({results['degradation_rate']:.1%})")
                print(f"   ➖ No changes: {results['no_changes']}")
                print(f"   ⏱️ Time: {results['elapsed_time_seconds']:.1f}s")
                
                # Add LLM timing statistics
                if 'llm_timing_stats' in results and results['llm_timing_stats']:
                    timing = results['llm_timing_stats']
                    print(f"   🤖 LLM timing: {timing['total_llm_calls']} calls")
                    print(f"      ⏱️ Avg decision: {timing['avg_decision_time']}s")
                    print(f"      ⏱️ Min/Max: {timing['min_decision_time']}s / {timing['max_decision_time']}s")
                    print(f"      ⏱️ Total LLM time: {timing['total_llm_time']}s")
                
                # Performance by rank analysis
                if 'results' in results:
                    rank_analysis = {}
                    for result in results['results']:
                        if result['status'] == 'success':
                            expected_rank = result['expected_rank']
                            if expected_rank not in rank_analysis:
                                rank_analysis[expected_rank] = {'total': 0, 'improved': 0}
                            rank_analysis[expected_rank]['total'] += 1
                            if result['improvement'] > 0:
                                rank_analysis[expected_rank]['improved'] += 1
                    
                    print(f"   � Performance by original rank:")
                    for rank in sorted(rank_analysis.keys()):
                        stats = rank_analysis[rank]
                        rate = stats['improved'] / stats['total'] if stats['total'] > 0 else 0
                        print(f"      Rank {rank}: {stats['improved']}/{stats['total']} improved ({rate:.1%})")
            else:
                print(f"\n🤖 {llm_model}: ❌ {results['status']}")
        
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Optimized LLM Voting Performance Evaluation")
    parser.add_argument('--dataset', required=True, 
                       choices=['Flickr', 'COCO', 'VizWiz'],
                       help='Dataset to evaluate (Flickr, COCO, or VizWiz)')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory (default: report/performance_vote/DATASET)')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = OptimizedPerformanceVoteEvaluator(args.dataset)
        
        # Run evaluation
        results = evaluator.run_full_evaluation()
        
        # Save results
        output_dir = Path(args.output_dir) if args.output_dir else None
        evaluator.save_results(results, output_dir)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
