"""
Performance Evaluation for CLIP-based Image Retrieval System

For each image in a VectorDB:
1. Embeds its caption using CLIP text encoder
2. Searches for most similar images in the vector database  
3. Records the rank of the original image
4. Generates statistics, visualizations, and detailed analysis
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from tqdm import tqdm
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import warnings
from datetime import datetime
import statistics
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndividualVisualizationGenerator:
    """
    Generates individual visualization files for each analysis aspect
    """
    def __init__(self, json_file, output_dir):
        self.json_file = json_file
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()
        
    def load_data(self):
        """Load performance data from JSON file"""
        logger.info(f"📁 Loading performance data from {self.json_file}")
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Check if data has the expected structure
        if 'ranks' in self.data:
            # New format with ranks array
            self.ranks = np.array(self.data['ranks'])
            logger.info(f"✅ Loaded {len(self.ranks):,} rank results (new format)")
        else:
            # Old format with list of items
            self.ranks = np.array([item['rank'] for item in self.data])
            logger.info(f"✅ Loaded {len(self.ranks):,} rank results (old format)")
        
    def create_rank_distribution(self):
        """Create rank distribution histogram"""
        plt.figure(figsize=(12, 8))
        
        # Calculate statistics
        mean_rank = np.mean(self.ranks)
        median_rank = np.median(self.ranks)
        
        # Create histogram with log scale
        bins = np.logspace(0, np.log10(max(self.ranks)), 50)
        plt.hist(self.ranks, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xscale('log')
        
        # Add mean and median lines
        plt.axvline(mean_rank, color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {mean_rank:.0f}")
        plt.axvline(median_rank, color='orange', linestyle='--', linewidth=2,
                   label=f"Median: {median_rank:.0f}")
        
        plt.xlabel('Rank (log scale)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('CLIP Retrieval: Rank Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self.output_dir / "01_rank_distribution.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def create_cumulative_accuracy(self):
        """Create cumulative accuracy curve"""
        plt.figure(figsize=(12, 8))
        
        # Calculate cumulative accuracy
        k_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        accuracies = []
        
        for k in k_values:
            accuracy = np.mean(self.ranks <= k) * 100
            accuracies.append(accuracy)
        
        # Plot
        plt.plot(k_values, accuracies, 'b-o', linewidth=3, markersize=8)
        plt.xscale('log')
        plt.xlabel('Top-K', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('CLIP Retrieval: Cumulative Top-K Accuracy', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key points
        key_points = [(1, accuracies[0]), (10, accuracies[3]), (100, accuracies[6])]
        for k, acc in key_points:
            plt.annotate(f'Top-{k}: {acc:.1f}%', 
                        xy=(k, acc), xytext=(k*2, acc-5),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / "02_cumulative_accuracy.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def create_quality_categories(self):
        """Create quality categories bar chart"""
        plt.figure(figsize=(12, 8))
        
        # Define quality categories
        categories = {
            'Perfect\\n(Rank = 1)': np.sum(self.ranks == 1),
            'Excellent\\n(Rank 2-10)': np.sum((self.ranks >= 2) & (self.ranks <= 10)),
            'Good\\n(Rank 11-100)': np.sum((self.ranks >= 11) & (self.ranks <= 100)),
            'Fair\\n(Rank 101-1000)': np.sum((self.ranks >= 101) & (self.ranks <= 1000)),
            'Poor\\n(Rank > 1000)': np.sum(self.ranks > 1000)
        }
        
        # Create bar chart
        colors = ['gold', 'lightgreen', 'skyblue', 'orange', 'lightcoral']
        bars = plt.bar(categories.keys(), categories.values(), color=colors, edgecolor='black')
        
        # Add percentage labels on bars
        total = len(self.ranks)
        for bar, count in zip(bars, categories.values()):
            height = bar.get_height()
            percentage = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + total*0.005,
                    f'{count:,}\\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.xlabel('Quality Category', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('CLIP Retrieval: Quality Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.output_dir / "03_quality_categories.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def create_percentiles_chart(self):
        """Create percentiles visualization"""
        plt.figure(figsize=(12, 8))
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        values = [np.percentile(self.ranks, p) for p in percentiles]
        
        # Create bar chart
        bars = plt.bar([f'{p}th' for p in percentiles], values, 
                      color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.xlabel('Percentile', fontsize=12)
        plt.ylabel('Rank', fontsize=12)
        plt.title('CLIP Retrieval: Rank Percentiles', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.output_dir / "04_percentiles.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def create_box_plot(self):
        """Create box plot for rank distribution"""
        plt.figure(figsize=(12, 8))
        
        # Create box plot
        bp = plt.boxplot(self.ranks, vert=True, patch_artist=True, 
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
        
        plt.ylabel('Rank (log scale)', fontsize=12)
        plt.title('CLIP Retrieval: Rank Distribution Box Plot', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {np.mean(self.ranks):.0f}
Median: {np.median(self.ranks):.0f}
Q1: {np.percentile(self.ranks, 25):.0f}
Q3: {np.percentile(self.ranks, 75):.0f}
Min: {np.min(self.ranks):.0f}
Max: {np.max(self.ranks):.0f}"""
        
        plt.text(1.1, np.mean(self.ranks), stats_text, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
                fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        filepath = self.output_dir / "05_box_plot.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def create_top_k_comparison(self):
        """Create Top-K accuracy comparison"""
        plt.figure(figsize=(12, 8))
        
        # Define K values
        k_values = [1, 5, 10, 20, 50, 100, 500, 1000]
        accuracies = []
        
        for k in k_values:
            accuracy = np.mean(self.ranks <= k) * 100
            accuracies.append(accuracy)
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
        bars = plt.bar([f'Top-{k}' for k in k_values], accuracies, 
                      color=colors, edgecolor='black')
        
        # Add percentage labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.xlabel('Top-K', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('CLIP Retrieval: Top-K Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.output_dir / "06_topk_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def create_detailed_breakdown(self):
        """Create detailed quality breakdown"""
        plt.figure(figsize=(12, 8))
        
        # More detailed categories
        categories = {
            'Perfect (1)': np.sum(self.ranks == 1),
            'Excellent (2-5)': np.sum((self.ranks >= 2) & (self.ranks <= 5)),
            'V.Good (6-10)': np.sum((self.ranks >= 6) & (self.ranks <= 10)),
            'Good (11-20)': np.sum((self.ranks >= 11) & (self.ranks <= 20)),
            'Good (21-50)': np.sum((self.ranks >= 21) & (self.ranks <= 50)),
            'Fair (51-100)': np.sum((self.ranks >= 51) & (self.ranks <= 100)),
            'Fair (101-200)': np.sum((self.ranks >= 101) & (self.ranks <= 200)),
            'Poor (201-500)': np.sum((self.ranks >= 201) & (self.ranks <= 500)),
            'Poor (501-1000)': np.sum((self.ranks >= 501) & (self.ranks <= 1000)),
            'V.Poor (1001+)': np.sum(self.ranks > 1000)
        }
        
        # Create horizontal bar chart
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(categories)))
        bars = plt.barh(list(categories.keys()), list(categories.values()), 
                       color=colors, edgecolor='black')
        
        # Add count and percentage labels
        total = len(self.ranks)
        for i, (bar, count) in enumerate(zip(bars, categories.values())):
            width = bar.get_width()
            percentage = (count / total) * 100
            plt.text(width + total*0.005, bar.get_y() + bar.get_height()/2,
                    f'{count:,} ({percentage:.1f}%)',
                    ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.xlabel('Number of Images', fontsize=12)
        plt.ylabel('Quality Category', fontsize=12)
        plt.title('CLIP Retrieval: Detailed Quality Breakdown', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        filepath = self.output_dir / "07_detailed_breakdown.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def create_rank_frequency_top20(self):
        """Create top 20 most frequent ranks"""
        plt.figure(figsize=(12, 8))
        
        # Get rank frequency
        rank_counts = Counter(self.ranks)
        top_20 = rank_counts.most_common(20)
        
        ranks, counts = zip(*top_20)
        
        # Create bar chart
        bars = plt.bar(range(len(ranks)), counts, color='mediumpurple', 
                      edgecolor='black', alpha=0.7)
        
        # Customize x-axis
        plt.xticks(range(len(ranks)), [str(r) for r in ranks], rotation=45)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('Rank', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('CLIP Retrieval: Top 20 Most Frequent Ranks', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = self.output_dir / "08_rank_frequency_top20.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Saved: {filepath}")
        
    def generate_all_visualizations(self):
        """Generate all individual visualizations"""
        logger.info("🚀 Starting individual visualization generation...")
        logger.info("=" * 60)
        
        # Generate each visualization
        self.create_rank_distribution()
        self.create_cumulative_accuracy()
        self.create_quality_categories()
        self.create_percentiles_chart()
        self.create_box_plot()
        self.create_top_k_comparison()
        self.create_detailed_breakdown()
        self.create_rank_frequency_top20()
        
        logger.info("✅ All individual visualizations generated successfully!")
        logger.info(f"📂 Files saved in: {self.output_dir}")
        
        # Generate summary statistics
        self.generate_summary_stats()
        
    def generate_summary_stats(self):
        """Generate summary statistics file"""
        stats = {
            "total_images": len(self.ranks),
            "basic_statistics": {
                "mean_rank": float(np.mean(self.ranks)),
                "median_rank": float(np.median(self.ranks)),
                "std_rank": float(np.std(self.ranks)),
                "min_rank": int(np.min(self.ranks)),
                "max_rank": int(np.max(self.ranks))
            },
            "percentiles": {
                f"{p}th": float(np.percentile(self.ranks, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
            "top_k_accuracy": {
                f"top_{k}": f"{np.mean(self.ranks <= k) * 100:.2f}%"
                for k in [1, 5, 10, 20, 50, 100, 500, 1000]
            },
            "quality_distribution": {
                "perfect": int(np.sum(self.ranks == 1)),
                "excellent_2_10": int(np.sum((self.ranks >= 2) & (self.ranks <= 10))),
                "good_11_100": int(np.sum((self.ranks >= 11) & (self.ranks <= 100))),
                "fair_101_1000": int(np.sum((self.ranks >= 101) & (self.ranks <= 1000))),
                "poor_1000_plus": int(np.sum(self.ranks > 1000))
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "00_summary_statistics.json"
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Saved summary: {summary_path}")
        
        # Print key stats
        logger.info("📈 Key Performance Metrics:")
        logger.info(f"   • Mean Rank: {stats['basic_statistics']['mean_rank']:.1f}")
        logger.info(f"   • Median Rank: {stats['basic_statistics']['median_rank']:.1f}")
        logger.info(f"   • Top-1 Accuracy: {stats['top_k_accuracy']['top_1']}")
        logger.info(f"   • Top-10 Accuracy: {stats['top_k_accuracy']['top_10']}")
        logger.info(f"   • Top-100 Accuracy: {stats['top_k_accuracy']['top_100']}")

class PerformanceEvaluator:
    """
    Comprehensive performance evaluator for CLIP-based image retrieval system.
    """
    
    def __init__(self, vectordb_name: str, vectordb_dir: str = "VectorDBs", output_dir: str = None):
        self.vectordb_name = vectordb_name
        self.vectordb_dir = Path(vectordb_dir).resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "openai/clip-vit-large-patch14"
        
        # Initialize output directory
        if output_dir is None:
            self.output_dir = Path(".")
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize paths
        self.results_file = self.output_dir / "performance.json"
        self.viz_dir = self.output_dir / "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # Set to evaluation mode and enable half precision if using GPU
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        
        # Database components
        self.index = None
        self.metadata = None
        self.total_images = 0
        
        logger.info("✅ Performance Evaluator initialized")
    
    def load_vectordb(self) -> bool:
        """Load the specified vector database"""
        try:
            index_file = self.vectordb_dir / f"{self.vectordb_name}.index"
            metadata_file = self.vectordb_dir / f"{self.vectordb_name}_metadata.json"
            
            if not index_file.exists() or not metadata_file.exists():
                logger.error(f"Database files not found for: {self.vectordb_name}")
                return False
            
            # Load FAISS index
            logger.info(f"Loading FAISS index: {index_file}")
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            logger.info(f"Loading metadata: {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                # Extract the metadata list from the dictionary
                self.metadata = metadata_dict.get("metadata", metadata_dict)
            
            self.total_images = len(self.metadata)
            
            logger.info(f"✅ Database '{self.vectordb_name}' loaded successfully")
            logger.info(f"   📊 {self.total_images:,} entries")
            logger.info(f"   🎯 Vector dimension: {self.index.d}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text using CLIP model"""
        try:
            # Process text
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text embedding
            with torch.no_grad():
                if self.device.type == "cuda":
                    # Use half precision for GPU
                    with torch.amp.autocast('cuda'):
                        text_features = self.model.get_text_features(**inputs)
                else:
                    text_features = self.model.get_text_features(**inputs)
                
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = text_features.cpu().numpy().astype(np.float32)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text '{text}': {e}")
            return None
    
    def find_image_rank(self, caption: str, target_image_id: int) -> int:
        """
        Find the exact rank of target image when searching with its caption
        Searches through ALL images until the target is found
        
        Args:
            caption: Text caption to search with
            target_image_id: Index of the target image in metadata
            
        Returns:
            Exact rank of the target image (1-indexed), or -1 if error occurs
        """
        # Encode caption
        query_embedding = self.encode_text(caption)
        if query_embedding is None:
            return -1
        
        # Search in FAISS index - get ALL results
        try:
            # Search through all images in the database
            distances, indices = self.index.search(query_embedding, self.total_images)
            
            # Find the rank of our target image
            for rank, idx in enumerate(indices[0], 1):
                if idx == target_image_id:
                    return rank
            
            # This should never happen if the image is in the database
            logger.error(f"Target image {target_image_id} not found in search results")
            return -1
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return -1
    
    def evaluate_performance(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate performance on all images or a sample
        Finds EXACT rank for each image by searching through ALL results
        
        Args:
            sample_size: Number of images to evaluate (None for all)
            
        Returns:
            Dictionary with performance metrics and results
        """
        if not self.index or not self.metadata:
            logger.error("Database not loaded")
            return {}
        
        # Determine which images to evaluate
        if sample_size and sample_size < self.total_images:
            # Random sample
            indices = np.random.choice(self.total_images, sample_size, replace=False)
            logger.info(f"Evaluating random sample of {sample_size} images")
        else:
            # All images
            indices = list(range(self.total_images))
            logger.info(f"Evaluating all {self.total_images} images - EXACT RANKS")
        
        results = {
            "vectordb_name": self.vectordb_name,
            "evaluation_date": datetime.now().isoformat(),
            "total_images_in_db": self.total_images,
            "images_evaluated": len(indices),
            "exact_rank_search": True,
            "max_possible_rank": self.total_images,
            "ranks": [],
            "image_results": [],
            "statistics": {},
            "examples": {
                "best": [],
                "worst": [],
                "random_good": [],
                "random_bad": []
            }
        }
        
        ranks = []
        successful_evaluations = 0
        
        # Evaluate each image
        logger.info("Starting performance evaluation with EXACT rank search...")
        logger.info("This will search through ALL images for each caption")
        
        for i, image_idx in enumerate(tqdm(indices, desc="Finding exact ranks")):
            try:
                metadata_entry = self.metadata[image_idx]
                caption = metadata_entry.get('caption', '')
                
                if not caption or len(caption.strip()) < 3:
                    continue
                
                # Find EXACT rank (no limit)
                rank = self.find_image_rank(caption, image_idx)
                
                if rank > 0:  # Successfully found
                    ranks.append(rank)
                    successful_evaluations += 1
                    
                    # Store detailed result
                    image_result = {
                        "image_id": image_idx,
                        "filename": metadata_entry.get('filename', 'unknown'),
                        "caption": caption,
                        "exact_rank": rank,
                        "found": True
                    }
                    
                    results["image_results"].append(image_result)
                    
                    # Collect examples based on exact rank
                    if rank == 1:
                        results["examples"]["best"].append(image_result)
                    elif rank <= 5:
                        results["examples"]["random_good"].append(image_result)
                    elif rank <= 100:
                        # Keep some medium examples
                        if len(results["examples"].get("medium", [])) < 20:
                            if "medium" not in results["examples"]:
                                results["examples"]["medium"] = []
                            results["examples"]["medium"].append(image_result)
                    elif rank > 1000:
                        results["examples"]["random_bad"].append(image_result)
                    
                    # Find the worst examples (highest ranks)
                    if len(results["examples"]["worst"]) < 20:
                        results["examples"]["worst"].append(image_result)
                    else:
                        # Replace if this rank is worse than the best in worst
                        min_worst_rank = min(ex["exact_rank"] for ex in results["examples"]["worst"])
                        if rank > min_worst_rank:
                            # Remove the best of the worst and add this one
                            results["examples"]["worst"] = [ex for ex in results["examples"]["worst"] 
                                                          if ex["exact_rank"] != min_worst_rank]
                            results["examples"]["worst"].append(image_result)
                
                else:  # Error occurred
                    image_result = {
                        "image_id": image_idx,
                        "filename": metadata_entry.get('filename', 'unknown'),
                        "caption": caption,
                        "exact_rank": -1,
                        "found": False,
                        "error": True
                    }
                    results["image_results"].append(image_result)
                
                # Save intermediate results every 1000 images
                if (i + 1) % 1000 == 0:
                    self._save_intermediate_results(results, ranks)
                    logger.info(f"Processed {i+1}/{len(indices)} images - Current mean rank: {np.mean(ranks):.1f}")
            
            except Exception as e:
                logger.error(f"Error evaluating image {image_idx}: {e}")
                continue
        
        # Calculate final statistics
        if ranks:
            results["ranks"] = ranks
            results["statistics"] = self._calculate_exact_statistics(ranks, self.total_images)
            
            logger.info(f"✅ Evaluation completed!")
            logger.info(f"   📊 Successfully evaluated: {successful_evaluations}/{len(indices)} images")
            logger.info(f"   🎯 Mean exact rank: {results['statistics']['mean_rank']:.1f}")
            logger.info(f"   📈 Median exact rank: {results['statistics']['median_rank']:.1f}")
            logger.info(f"   🏆 Top-1 accuracy: {results['statistics']['top1_accuracy']:.2%}")
            logger.info(f"   🏅 Top-5 accuracy: {results['statistics']['top5_accuracy']:.2%}")
            logger.info(f"   📊 Best rank: {results['statistics']['min_rank']}")
            logger.info(f"   📊 Worst rank: {results['statistics']['max_rank']}")
        else:
            logger.warning("No successful evaluations completed")
        
        # Sort examples by rank for better analysis
        for key in results["examples"]:
            if key == "best":
                continue  # Already rank 1
            results["examples"][key] = sorted(results["examples"][key], 
                                            key=lambda x: x.get("exact_rank", float('inf')))[:20]
        
        return results
    
    def _calculate_exact_statistics(self, ranks: List[int], max_possible_rank: int) -> Dict[str, Any]:
        """Calculate comprehensive statistics from exact ranks"""
        ranks_array = np.array(ranks)
        
        # Basic statistics
        stats = {
            "count": len(ranks),
            "mean_rank": float(np.mean(ranks_array)),
            "median_rank": float(np.median(ranks_array)),
            "std_rank": float(np.std(ranks_array)),
            "min_rank": int(np.min(ranks_array)),
            "max_rank": int(np.max(ranks_array)),
            "max_possible_rank": max_possible_rank
        }
        
        # Top-K accuracies (more comprehensive since we have exact ranks)
        top_k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        for k in top_k_values:
            if k <= max_possible_rank:
                accuracy = np.mean(ranks_array <= k)
                stats[f"top{k}_accuracy"] = float(accuracy)
        
        # Percentiles (more detailed)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        for p in percentiles:
            stats[f"p{p}_rank"] = float(np.percentile(ranks_array, p))
        
        # Distribution analysis (more granular for exact ranks)
        stats["rank_distribution"] = {
            "1": float(np.mean(ranks_array == 1)),
            "2-5": float(np.mean((ranks_array >= 2) & (ranks_array <= 5))),
            "6-10": float(np.mean((ranks_array >= 6) & (ranks_array <= 10))),
            "11-50": float(np.mean((ranks_array >= 11) & (ranks_array <= 50))),
            "51-100": float(np.mean((ranks_array >= 51) & (ranks_array <= 100))),
            "101-500": float(np.mean((ranks_array >= 101) & (ranks_array <= 500))),
            "501-1000": float(np.mean((ranks_array >= 501) & (ranks_array <= 1000))),
            "1001-5000": float(np.mean((ranks_array >= 1001) & (ranks_array <= 5000))),
            "5001-10000": float(np.mean((ranks_array >= 5001) & (ranks_array <= 10000))),
            "10000+": float(np.mean(ranks_array > 10000))
        }
        
        # Rank quality analysis
        stats["quality_analysis"] = {
            "excellent": float(np.mean(ranks_array <= 10)),  # Top 10
            "very_good": float(np.mean(ranks_array <= 100)),  # Top 100
            "good": float(np.mean(ranks_array <= 1000)),  # Top 1000
            "fair": float(np.mean(ranks_array <= 10000)),  # Top 10k
            "poor": float(np.mean(ranks_array > 10000))  # Beyond top 10k
        }
        
        return stats
    
    def _save_intermediate_results(self, results: Dict, ranks: List[int]):
        """Save intermediate results to avoid data loss"""
        try:
            # Update statistics with current ranks
            if ranks:
                results["ranks"] = ranks
                results["statistics"] = self._calculate_exact_statistics(ranks, self.total_images)
            
            # Save to file
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Intermediate results saved to {self.results_file}")
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save final results to JSON file"""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results saved to {self.results_file}")
            logger.info(f"   📊 File size: {self.results_file.stat().st_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive visualizations"""
        if not results.get("ranks"):
            logger.warning("No ranks data available for visualization")
            return
        
        ranks = np.array(results["ranks"])
        stats = results["statistics"]
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Rank Distribution Histogram
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(ranks, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(stats['mean_rank'], color='red', linestyle='--', label=f'Mean: {stats["mean_rank"]:.1f}')
        plt.axvline(stats['median_rank'], color='orange', linestyle='--', label=f'Median: {stats["median_rank"]:.1f}')
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title('Distribution of Image Ranks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Log-scale Histogram
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(ranks, bins=np.logspace(0, np.log10(max(ranks)), 50), alpha=0.7, edgecolor='black')
        plt.xscale('log')
        plt.xlabel('Rank (log scale)')
        plt.ylabel('Frequency')
        plt.title('Rank Distribution (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # 3. Cumulative Distribution Function
        ax3 = plt.subplot(2, 3, 3)
        sorted_ranks = np.sort(ranks)
        y_values = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        plt.plot(sorted_ranks, y_values, linewidth=2)
        plt.xlabel('Rank')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Ranks')
        plt.grid(True, alpha=0.3)
        
        # Add percentile lines
        for p in [50, 75, 90, 95]:
            rank_p = stats[f'p{p}_rank']
            plt.axvline(rank_p, linestyle=':', alpha=0.7, label=f'P{p}: {rank_p:.0f}')
        plt.legend()
        
        # 4. Top-K Performance
        ax4 = plt.subplot(2, 3, 4)
        k_values = [1, 5, 10, 20, 50, 100]
        accuracies = [stats.get(f'top{k}_accuracy', 0) for k in k_values]
        
        plt.bar(range(len(k_values)), accuracies, alpha=0.7)
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('Top-K Accuracy Performance')
        plt.xticks(range(len(k_values)), [f'Top-{k}' for k in k_values], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add accuracy labels on bars
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.2%}', ha='center', va='bottom')
        
        # 5. Rank Distribution by Categories
        ax5 = plt.subplot(2, 3, 5)
        categories = ['1', '2-5', '6-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5001-10000', '10000+']
        # Filter only categories that exist in stats and have non-zero values
        available_categories = []
        cat_values = []
        for cat in categories:
            if cat in stats['rank_distribution'] and stats['rank_distribution'][cat] > 0:
                available_categories.append(cat)
                cat_values.append(stats['rank_distribution'][cat] * 100)  # Convert to percentage
        
        # Use only the first 6 categories to fit the display
        available_categories = available_categories[:6]
        cat_values = cat_values[:6]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(available_categories)))
        bars = plt.bar(available_categories, cat_values, color=colors, alpha=0.7)
        plt.xlabel('Rank Range')
        plt.ylabel('Percentage of Images')
        plt.title('Distribution by Rank Categories')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, val in zip(bars, cat_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom')
        
        # 6. Box Plot
        ax6 = plt.subplot(2, 3, 6)
        box_data = [ranks[ranks <= 10], ranks[(ranks > 10) & (ranks <= 100)], ranks[ranks > 100]]
        box_labels = ['Ranks 1-10', 'Ranks 11-100', 'Ranks >100']
        
        bp = plt.boxplot([data for data in box_data if len(data) > 0], 
                        labels=[label for data, label in zip(box_data, box_labels) if len(data) > 0],
                        patch_artist=True)
        
        # Color the boxes
        colors = ['lightgreen', 'yellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel('Rank')
        plt.title('Rank Distribution by Performance Level')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add overall title and metadata
        fig.suptitle(f'Performance Analysis: {results["vectordb_name"]}\n'
                    f'Evaluated {len(ranks):,} images | Mean Rank: {stats["mean_rank"]:.1f} | '
                    f'Top-1: {stats["top1_accuracy"]:.1%} | Top-5: {stats["top5_accuracy"]:.1%}',
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.viz_dir / "performance_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualizations saved to {plot_file}")
        
        plt.show()
        
        # Create a second figure for detailed examples
        self._create_examples_visualization(results)
    
    def _create_examples_visualization(self, results: Dict[str, Any]):
        """Create visualization showing best and worst examples"""
        examples = results.get("examples", {})
        
        if not any(examples.values()):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Examples Analysis', fontsize=16)
        
        # Best Examples (Rank 1)
        ax1 = axes[0, 0]
        best_examples = examples.get("best", [])[:10]
        if best_examples:
            captions = [ex["caption"][:50] + "..." if len(ex["caption"]) > 50 else ex["caption"] 
                       for ex in best_examples]
            ranks = [ex.get("exact_rank", ex.get("rank", 1)) for ex in best_examples]
            
            bars = ax1.barh(range(len(captions)), [1] * len(captions), color='green', alpha=0.7)
            ax1.set_yticks(range(len(captions)))
            ax1.set_yticklabels(captions, fontsize=8)
            ax1.set_xlabel('Rank')
            ax1.set_title(f'Perfect Matches (Rank 1) - {len(best_examples)} examples')
            ax1.grid(True, alpha=0.3)
        
        # Good Examples (Rank 2-5)
        ax2 = axes[0, 1]
        good_examples = examples.get("random_good", [])[:10]
        if good_examples:
            captions = [ex["caption"][:50] + "..." if len(ex["caption"]) > 50 else ex["caption"] 
                       for ex in good_examples]
            ranks = [ex.get("exact_rank", ex.get("rank", 1)) for ex in good_examples]
            
            bars = ax2.barh(range(len(captions)), ranks, color='orange', alpha=0.7)
            ax2.set_yticks(range(len(captions)))
            ax2.set_yticklabels(captions, fontsize=8)
            ax2.set_xlabel('Rank')
            ax2.set_title(f'Good Matches (Rank 2-5) - {len(good_examples)} examples')
            ax2.grid(True, alpha=0.3)
            
            # Add rank labels on bars
            for i, rank in enumerate(ranks):
                ax2.text(rank + 0.1, i, str(rank), va='center')
        
        # Bad Examples (Rank > 100)
        ax3 = axes[1, 0]
        bad_examples = examples.get("random_bad", [])[:10]
        if bad_examples:
            captions = [ex["caption"][:50] + "..." if len(ex["caption"]) > 50 else ex["caption"] 
                       for ex in bad_examples]
            ranks = [ex.get("exact_rank", ex.get("rank", 1)) for ex in bad_examples]
            
            bars = ax3.barh(range(len(captions)), ranks, color='red', alpha=0.7)
            ax3.set_yticks(range(len(captions)))
            ax3.set_yticklabels(captions, fontsize=8)
            ax3.set_xlabel('Rank')
            ax3.set_title(f'Poor Matches (Rank > 100) - {len(bad_examples)} examples')
            ax3.grid(True, alpha=0.3)
            
            # Add rank labels on bars
            for i, rank in enumerate(ranks):
                ax3.text(rank + max(ranks) * 0.01, i, str(rank), va='center')
        
        # Worst Examples (Not found in top-1000)
        ax4 = axes[1, 1]
        worst_examples = examples.get("worst", [])[:10]
        if worst_examples:
            captions = [ex["caption"][:50] + "..." if len(ex["caption"]) > 50 else ex["caption"] 
                       for ex in worst_examples]
            
            bars = ax4.barh(range(len(captions)), [1000] * len(captions), color='darkred', alpha=0.7)
            ax4.set_yticks(range(len(captions)))
            ax4.set_yticklabels(captions, fontsize=8)
            ax4.set_xlabel('Rank')
            ax4.set_title(f'Failed Matches (Rank > 1000) - {len(worst_examples)} examples')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the examples plot
        examples_plot_file = self.viz_dir / "performance_examples.png"
        plt.savefig(examples_plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Examples visualization saved to {examples_plot_file}")
        
        plt.show()
    
    def run_evaluation(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Run complete evaluation pipeline with EXACT rank search"""
        logger.info("🚀 Starting Performance Evaluation - EXACT RANKS")
        logger.info("=" * 60)
        
        # Load database
        if not self.load_vectordb():
            return {}
        
        # Run evaluation
        results = self.evaluate_performance(sample_size)
        
        if results:
            # Save results
            self.save_results(results)
            
            # Create visualizations
            self.create_visualizations(results)
            
            # Create individual visualizations
            self.create_individual_visualizations(generate_individual=True)
            
            # Print summary
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of results"""
        stats = results.get("statistics", {})
        examples = results.get("examples", {})
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"🗂️  Database: {results['vectordb_name']}")
        print(f"📅 Date: {results['evaluation_date']}")
        print(f"🔢 Images evaluated: {results['images_evaluated']:,} / {results['total_images_in_db']:,}")
        
        if stats:
            print("\n📈 RANKING STATISTICS:")
            print("-" * 30)
            print(f"Mean rank: {stats['mean_rank']:.1f}")
            print(f"Median rank: {stats['median_rank']:.1f}")
            print(f"Standard deviation: {stats['std_rank']:.1f}")
            print(f"Best rank: {stats['min_rank']}")
            print(f"Worst rank: {stats['max_rank']}")
            print(f"Max possible rank: {stats.get('max_possible_rank', 'N/A')}")
            
            print("\n🏆 TOP-K ACCURACY:")
            print("-" * 30)
            for k in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
                acc_key = f"top{k}_accuracy"
                if acc_key in stats:
                    print(f"Top-{k:5d}: {stats[acc_key]:.3%}")
            
            print("\n📊 RANK DISTRIBUTION:")
            print("-" * 30)
            dist = stats.get('rank_distribution', {})
            for range_name, proportion in dist.items():
                print(f"Rank {range_name:>10}: {proportion:.2%}")
            
            print("\n📍 PERCENTILES:")
            print("-" * 30)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
                percentile_key = f"p{p}_rank"
                if percentile_key in stats:
                    print(f"P{p:4.1f}: {stats[percentile_key]:.0f}")
            
            # Quality analysis if available
            quality = stats.get('quality_analysis', {})
            if quality:
                print("\n🎯 QUALITY ANALYSIS:")
                print("-" * 30)
                print(f"Excellent (≤10):   {quality.get('excellent', 0):.2%}")
                print(f"Very Good (≤100):  {quality.get('very_good', 0):.2%}")
                print(f"Good (≤1000):      {quality.get('good', 0):.2%}")
                print(f"Fair (≤10000):     {quality.get('fair', 0):.2%}")
                print(f"Poor (>10000):     {quality.get('poor', 0):.2%}")
        
        if examples:
            print("\n🎯 EXAMPLES:")
            print("-" * 30)
            print(f"Perfect matches (rank 1): {len(examples.get('best', []))}")
            print(f"Good matches (rank 2-5): {len(examples.get('random_good', []))}")
            print(f"Poor matches (rank >100): {len(examples.get('random_bad', []))}")
            print(f"Failed matches (rank >1000): {len(examples.get('worst', []))}")
            
            # Show a few examples
            if examples.get('best'):
                print("\n✅ PERFECT MATCHES (Rank 1):")
                for i, ex in enumerate(examples['best'][:3]):
                    print(f"   {i+1}. {ex['filename']}: '{ex['caption'][:80]}...'")
            
            if examples.get('worst'):
                print("\n❌ FAILED MATCHES (Rank >1000):")
                for i, ex in enumerate(examples['worst'][:3]):
                    print(f"   {i+1}. {ex['filename']}: '{ex['caption'][:80]}...'")
        
        print("\n" + "=" * 60)
        print("📁 Files created:")
        print(f"   💾 Results: {self.results_file}")
        print(f"   📊 Main visualizations: {self.viz_dir}")
        print(f"   � Individual visualizations: {self.viz_dir / 'individual'}")
        print("=" * 60)
    
    def create_individual_visualizations(self, generate_individual=True):
        """Create individual visualization files for detailed analysis"""
        if not generate_individual:
            return
            
        logger.info("🎨 Generating individual visualizations...")
        
        # Create individual visualizations subdirectory
        individual_viz_dir = self.viz_dir / "individual"
        os.makedirs(individual_viz_dir, exist_ok=True)
        
        # Generate individual visualizations
        viz_generator = IndividualVisualizationGenerator(str(self.results_file), str(individual_viz_dir))
        viz_generator.generate_all_visualizations()
        
        logger.info(f"✅ Individual visualizations saved to: {individual_viz_dir}")

def load_and_analyze_existing_results(results_file: str = "performance.json"):
    """Load and analyze existing performance results"""
    results_path = Path(results_file)
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    logger.info(f"Loading existing results from {results_file}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Create a temporary evaluator for visualization methods
    evaluator = PerformanceEvaluator("temp")
    
    # Create visualizations
    evaluator.create_visualizations(results)
    
    # Print summary
    evaluator._print_summary(results)
    
    return results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Performance Evaluation for CLIP-based Image Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--vectordb',
        type=str,
        required=True,
        help='Name of the VectorDB to evaluate (e.g., COCO_VectorDB)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of images to evaluate (default: all images)'
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing results without running new evaluation'
    )
    
    parser.add_argument(
        '--results-file',
        type=str,
        default='performance.json',
        help='Results file to analyze (when using --analyze-only)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save performance results and visualizations (default: current directory)'
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        load_and_analyze_existing_results(args.results_file)
    else:
        # Run new evaluation
        evaluator = PerformanceEvaluator(args.vectordb, output_dir=args.output_dir)
        results = evaluator.run_evaluation(args.sample_size)
        
        if results:
            logger.info("🎉 Evaluation completed successfully!")
        else:
            logger.error("❌ Evaluation failed!")

if __name__ == "__main__":
    main()