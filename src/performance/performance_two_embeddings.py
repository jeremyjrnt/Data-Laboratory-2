"""
Performance Evaluation for Two Embeddings Retrieval System
Evaluates different systems and fusion methods on Flickr dataset
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retriever_two_embeddings import TwoEmbeddingsRetriever
from config.config import Config


class TwoEmbeddingsPerformanceEvaluator:
    """
    Evaluates performance of two embeddings retrieval systems
    """
    
    def __init__(self, dataset_name='Flickr', k=None):
        """
        Initialize the evaluator
        
        Args:
            dataset_name: Name of the dataset (default: Flickr)
            k: Number of results to retrieve (default: from Config)
        """
        self.dataset_name = dataset_name
        self.k = k if k is not None else Config.DEFAULT_K
        
        # Check GPU availability
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Paths
        self.base_path = Config.PROJECT_ROOT
        self.data_path = Config.get_dataset_dir(dataset_name)
        self.metadata_path = Config.get_metadata_path(dataset_name)
        self.output_path = Config.REPORT_DIR / "performance_double_embeddings" / dataset_name
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize retriever
        self.retriever = TwoEmbeddingsRetriever(dataset_name=dataset_name, k=k)
        
        # Load metadata
        self.metadata = []
        self._load_metadata()
        
        # Store current results for incremental saving
        self.all_results = None
        
    def _load_metadata(self):
        """Load metadata with images from selected_1000.json"""
        # Load from selected_1000.json instead of full metadata
        selected_path = self.data_path / "selected_1000.json"
        print(f"📂 Loading metadata from {selected_path}...")
        
        with open(selected_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'images' in data:
            self.metadata = data['images']
        elif isinstance(data, list):
            self.metadata = data
        else:
            raise ValueError("Unexpected metadata format")
        
        print(f"✓ Loaded {len(self.metadata)} images")
    
    def _save_incremental_results(self, output_file: Path, current_system_results: Dict):
        """Save results incrementally every 100 images"""
        if self.all_results is None:
            return
        
        # Update current system results in all_results
        for i, eval_result in enumerate(self.all_results['evaluations']):
            if eval_result['system_name'] == current_system_results['system_name']:
                self.all_results['evaluations'][i] = current_system_results
                break
        else:
            # If not found, add it
            self.all_results['evaluations'].append(current_system_results)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        
    def evaluate_system(self, systems: List[str], fusion_methods: List[str] = None, k_rrf: int = 60, output_file: Path = None) -> List[Dict]:
        """
        Evaluate system configuration(s) - if multiple fusion methods, compute all in one pass
        
        Args:
            systems: List of systems to use
            fusion_methods: List of fusion methods (if multiple systems) - compute all at once
            k_rrf: k parameter for RRF
            output_file: Path to save incremental results
            
        Returns:
            List of dictionaries with evaluation results (one per fusion method)
        """
        # If no fusion methods or single system, process as before
        if not fusion_methods:
            fusion_methods = [None]
        
        # Initialize results for each fusion method
        all_method_results = []
        for fusion_method in fusion_methods:
            system_name = "_".join(systems)
            if fusion_method:
                system_name += f"_{fusion_method}"
            
            # For single system, use 'single' instead of null for clarity
            display_fusion = fusion_method if fusion_method else 'single'
            
            all_method_results.append({
                'system_name': system_name,
                'systems': systems,
                'fusion_method': display_fusion,
                'k': self.k,
                'k_rrf': k_rrf if fusion_method == 'rrf' else None,
                'timestamp': datetime.now().isoformat(),
                'images': []
            })
        
        system_desc = "_".join(systems)
        if len(fusion_methods) > 1:
            system_desc += f" ({', '.join(fusion_methods)})"
        elif fusion_methods[0]:
            system_desc += f"_{fusion_methods[0]}"
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {system_desc}")
        print('='*70)
        
        # Process each image
        for idx, item in enumerate(tqdm(self.metadata, desc=f"Processing {system_desc}", unit="img")):
            filename = item.get('filename') or item.get('file_name') or item.get('image')
            baseline_rank = item.get('baseline_rank')
            
            if not filename or not baseline_rank:
                continue
            
            try:
                # Perform retrieval for ALL fusion methods at once (suppress output)
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    # Retrieve results for each fusion method
                    for method_idx, fusion_method in enumerate(fusion_methods):
                        result = self.retriever.retrieve(
                            filename=filename,
                            systems=systems,
                            fusion_method=fusion_method,
                            k_rrf=k_rrf
                        )
                        
                        new_rank = result['new_rank']
                        
                        if new_rank is None:
                            continue
                        
                        # Calculate progression
                        progression = baseline_rank - new_rank
                        
                        # Store result for this fusion method
                        image_result = {
                            'filename': filename,
                            'baseline_rank': baseline_rank,
                            'new_rank': new_rank,
                            'progression': progression
                        }
                        
                        all_method_results[method_idx]['images'].append(image_result)
                
                # Save every 10 images (for all methods)
                if output_file and (len(all_method_results[0]['images']) % 10 == 0):
                    for method_result in all_method_results:
                        self._save_incremental_results(output_file, method_result)
                
            except Exception as e:
                print(f"\n⚠️  Error processing {filename}: {e}")
                continue
        
        # Calculate statistics for each fusion method
        for method_result in all_method_results:
            if method_result['images']:
                progressions = [img['progression'] for img in method_result['images']]
                baseline_ranks = [img['baseline_rank'] for img in method_result['images']]
                new_ranks = [img['new_rank'] for img in method_result['images']]
                
                method_result['statistics'] = {
                    'total_images': len(method_result['images']),
                    'mean_baseline_rank': sum(baseline_ranks) / len(baseline_ranks),
                    'mean_new_rank': sum(new_ranks) / len(new_ranks),
                    'mean_progression': sum(progressions) / len(progressions),
                    'improved': len([p for p in progressions if p > 0]),
                    'degraded': len([p for p in progressions if p < 0]),
                    'unchanged': len([p for p in progressions if p == 0])
                }
                
                print(f"\n📊 Statistics for {method_result['system_name']}:")
                print(f"   Total images: {method_result['statistics']['total_images']}")
                print(f"   Mean baseline rank: {method_result['statistics']['mean_baseline_rank']:.2f}")
                print(f"   Mean new rank: {method_result['statistics']['mean_new_rank']:.2f}")
                print(f"   Mean progression: {method_result['statistics']['mean_progression']:.2f}")
                print(f"   Improved: {method_result['statistics']['improved']} ({method_result['statistics']['improved']/method_result['statistics']['total_images']*100:.1f}%)")
                print(f"   Degraded: {method_result['statistics']['degraded']} ({method_result['statistics']['degraded']/method_result['statistics']['total_images']*100:.1f}%)")
                print(f"   Unchanged: {method_result['statistics']['unchanged']}")
        
        return all_method_results
        
    def evaluate_all(self):
        """
        Evaluate all system configurations with incremental saving
        """
        print(f"\n{'='*70}")
        print(f"Two Embeddings Performance Evaluation - {self.dataset_name}")
        print('='*70)
        
        # Create output file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_path / f"performance_two_embeddings_{timestamp}.json"
        
        # Initialize results structure
        self.all_results = {
            'dataset': self.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'k': self.k,
            'evaluations': []
        }
        
        # Save initial structure
        print(f"📁 Results will be saved to: {output_file}\n")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        
        # System configurations to evaluate (optimized to compute fusion methods together)
        configurations = [
            # Single system
            {
                'systems': ['average'],
                'fusion_methods': [None]
            },
            # Classic + BLIP with all fusion methods (computed in one pass)
            {
                'systems': ['classic', 'blip'],
                'fusion_methods': ['combsum', 'rrf', 'borda']
            },
            # Classic + BM25 with all fusion methods (computed in one pass)
            {
                'systems': ['classic', 'bm25'],
                'fusion_methods': ['combsum', 'rrf', 'borda']
            },
            # Average + BM25 with all fusion methods (computed in one pass)
            {
                'systems': ['average', 'bm25'],
                'fusion_methods': ['combsum', 'rrf', 'borda']
            },
            # Classic + BLIP + BM25 with all fusion methods (computed in one pass)
            {
                'systems': ['classic', 'blip', 'bm25'],
                'fusion_methods': ['combsum', 'rrf', 'borda']
            }
        ]
        
        # Evaluate each configuration (returns list of results for multiple fusion methods)
        for config in configurations:
            results_list = self.evaluate_system(
                systems=config['systems'],
                fusion_methods=config['fusion_methods'],
                output_file=output_file
            )
            
            # Results are already added via _save_incremental_results, just save final version
            print(f"💾 Saving final results for this configuration...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, indent=2, ensure_ascii=False)
            print(f"✓ Results saved to {output_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("📊 SUMMARY")
        print('='*70)
        
        for eval_result in self.all_results['evaluations']:
            if 'statistics' in eval_result:
                stats = eval_result['statistics']
                print(f"\n{eval_result['system_name']}:")
                print(f"   Mean progression: {stats['mean_progression']:+.2f}")
                print(f"   Mean rank: {stats['mean_baseline_rank']:.2f} → {stats['mean_new_rank']:.2f}")
                print(f"   Improved/Degraded/Unchanged: {stats['improved']}/{stats['degraded']}/{stats['unchanged']}")
        
        print(f"\n{'='*70}")
        print(f"✓ Evaluation completed!")
        print(f"📁 Final results saved to: {output_file}")
        print(f"{'='*70}")
        
        return self.all_results


def main():
    """Main function"""
    evaluator = TwoEmbeddingsPerformanceEvaluator(dataset_name='Flickr', k=None)
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
