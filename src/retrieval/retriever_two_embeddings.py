"""
FAISS Double Embedding Retriever

This module provides retrieval methods for the double embedding index:
- Early Fusion: Search on pre-computed average embeddings, text embeddings, or BM25
- Late Fusion: Combine multiple retrieval systems with various fusion methods

Fusion Methods:
- CombSum (weighted)
- Max-pooling
- RRF (Reciprocal Rank Fusion)
- Borda Count
- Minimax-rank
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import time

import numpy as np
import torch
import faiss
from transformers import CLIPModel, CLIPProcessor
from rank_bm25 import BM25Okapi

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DoubleEmbeddingRetriever:
    """
    Retriever for double embedding FAISS index with multiple fusion strategies
    """
    
    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 device: Optional[str] = None):
        """
        Initialize Double Embedding Retriever
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            clip_model_name: HuggingFace CLIP model name
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.clip_model_name = clip_model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        self._load_clip_model()
        
        # Load index and metadata
        self._load_index()
        
        # Initialize BM25
        self._initialize_bm25()
        
        logger.info("Double Embedding Retriever initialized successfully")
    
    def _load_clip_model(self):
        """Load CLIP model for text encoding"""
        try:
            logger.info(f"Loading CLIP model: {self.clip_model_name}")
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _load_index(self):
        """Load FAISS index and metadata"""
        try:
            # Load FAISS index
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index file not found: {self.index_path}")
            
            logger.info(f"Loading FAISS index from: {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
            logger.info(f"Loading metadata from: {self.metadata_path}")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_info = json.load(f)
            
            self.metadata = metadata_info['metadata']
            self.embedding_dim = metadata_info['embedding_dim']
            
            # Create index by embedding type for faster filtering
            self.metadata_by_type = {
                'image': [],
                'blip_text': [],
                'average': []
            }
            self.indices_by_type = {
                'image': [],
                'blip_text': [],
                'average': []
            }
            
            for idx, meta in enumerate(self.metadata):
                emb_type = meta.get('embedding_type', 'unknown')
                if emb_type in self.metadata_by_type:
                    self.metadata_by_type[emb_type].append(meta)
                    self.indices_by_type[emb_type].append(idx)
            
            logger.info(f"Loaded {len(self.metadata)} vectors")
            logger.info(f"  - Image embeddings: {len(self.indices_by_type['image'])}")
            logger.info(f"  - BLIP text embeddings: {len(self.indices_by_type['blip_text'])}")
            logger.info(f"  - Average embeddings: {len(self.indices_by_type['average'])}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def _initialize_bm25(self):
        """Initialize BM25 index on BLIP captions"""
        try:
            logger.info("Initializing BM25 index on BLIP captions...")
            
            # Collect all unique BLIP captions (one per image)
            # Use average type as reference (one entry per image)
            self.bm25_documents = []
            self.bm25_metadata = []
            
            for meta in self.metadata_by_type['average']:
                blip_caption = meta.get('blip_caption', '')
                if blip_caption:
                    # Tokenize (simple split by space, lowercase)
                    tokenized = blip_caption.lower().split()
                    self.bm25_documents.append(tokenized)
                    self.bm25_metadata.append(meta)
            
            # Create BM25 index
            if self.bm25_documents:
                self.bm25 = BM25Okapi(self.bm25_documents)
                logger.info(f"BM25 index created with {len(self.bm25_documents)} documents")
            else:
                self.bm25 = None
                logger.warning("No BLIP captions found for BM25")
                
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            self.bm25 = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query using CLIP
        
        Args:
            text: Text query
            
        Returns:
            Text embedding (normalized)
        """
        try:
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            return text_features.cpu().float().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def get_caption_for_image(self, image_identifier: str) -> Optional[str]:
        """
        Get the true caption for an image from metadata
        
        Args:
            image_identifier: Image ID or filename
            
        Returns:
            Caption string or None if not found
        """
        # Search in average metadata (one per image)
        for meta in self.metadata_by_type['average']:
            image_id = meta.get('image_id', '')
            image_path = meta.get('image_path', '')
            
            # Match by image_id or filename
            if (image_identifier == image_id or 
                image_identifier == os.path.basename(image_path) or
                image_identifier in image_path):
                return meta.get('caption', None)
        
        logger.warning(f"Caption not found for image: {image_identifier}")
        return None
    
    # ==================== EARLY FUSION METHODS ====================
    
    def search_early_fusion_average(self, query: str, k: int = 10) -> List[Dict]:
        """
        Early Fusion: Search on pre-computed average embeddings
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of results with metadata and scores
        """
        logger.info(f"Early Fusion (Average) search for: '{query}'")
        
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Search in full index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            len(self.metadata)  # Get all to filter
        )
        
        # Filter for average type only
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx]
            if meta.get('embedding_type') == 'average':
                results.append({
                    'image_id': meta.get('image_id'),
                    'image_path': meta.get('image_path'),
                    'caption': meta.get('caption'),
                    'blip_caption': meta.get('blip_caption'),
                    'similarity': float(dist),
                    'embedding_type': 'average'
                })
                
                if len(results) >= k:
                    break
        
        return results
    
    def search_early_fusion_blip_text(self, query: str, k: int = 10) -> List[Dict]:
        """
        Early Fusion: Search on BLIP text embeddings
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of results with metadata and scores
        """
        logger.info(f"Early Fusion (BLIP Text) search for: '{query}'")
        
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Search in full index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            len(self.metadata)
        )
        
        # Filter for blip_text type only
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx]
            if meta.get('embedding_type') == 'blip_text':
                results.append({
                    'image_id': meta.get('image_id'),
                    'image_path': meta.get('image_path'),
                    'caption': meta.get('caption'),
                    'blip_caption': meta.get('blip_caption'),
                    'similarity': float(dist),
                    'embedding_type': 'blip_text'
                })
                
                if len(results) >= k:
                    break
        
        return results
    
    def search_bm25(self, query: str, k: int = 10) -> List[Dict]:
        """
        BM25 search on BLIP captions
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of results with metadata and BM25 scores
        """
        logger.info(f"BM25 search for: '{query}'")
        
        if self.bm25 is None:
            logger.error("BM25 index not initialized")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1][:k]
        
        # Build results
        results = []
        for idx in sorted_indices:
            meta = self.bm25_metadata[idx]
            results.append({
                'image_id': meta.get('image_id'),
                'image_path': meta.get('image_path'),
                'caption': meta.get('caption'),
                'blip_caption': meta.get('blip_caption'),
                'similarity': float(scores[idx]),  # BM25 score
                'embedding_type': 'bm25'
            })
        
        return results
    
    # ==================== LATE FUSION METHODS ====================
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        Normalize similarity scores to [0, 1] range using min-max normalization
        
        Args:
            results: List of results with similarity scores
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        scores = [r['similarity'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for r in results:
                r['normalized_similarity'] = 1.0
        else:
            for r in results:
                r['normalized_similarity'] = (r['similarity'] - min_score) / (max_score - min_score)
        
        return results
    
    def _get_unique_images(self, results: List[Dict]) -> List[str]:
        """Get list of unique image IDs from results"""
        image_ids = []
        seen = set()
        for r in results:
            img_id = r['image_id']
            if img_id not in seen:
                image_ids.append(img_id)
                seen.add(img_id)
        return image_ids
    
    def late_fusion_combsum(self, 
                           query: str, 
                           systems: List[str],
                           weights: Optional[List[float]] = None,
                           k: int = 10,
                           k_per_system: int = 100) -> List[Dict]:
        """
        Late Fusion: Weighted CombSum
        
        Args:
            query: Text query
            systems: List of systems to combine ['image', 'blip_text', 'bm25']
            weights: Weights for each system (default: equal weights)
            k: Number of final results
            k_per_system: Number of results to retrieve from each system
            
        Returns:
            Fused results sorted by combined score
        """
        logger.info(f"Late Fusion CombSum with systems: {systems}")
        
        if weights is None:
            weights = [1.0 / len(systems)] * len(systems)
        
        if len(weights) != len(systems):
            raise ValueError("Number of weights must match number of systems")
        
        # Retrieve from each system
        system_results = []
        for system in systems:
            if system == 'image':
                # Search on image embeddings
                query_embedding = self.encode_text(query)
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype('float32'), 
                    len(self.metadata)
                )
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    meta = self.metadata[idx]
                    if meta.get('embedding_type') == 'image':
                        results.append({
                            'image_id': meta.get('image_id'),
                            'image_path': meta.get('image_path'),
                            'caption': meta.get('caption'),
                            'blip_caption': meta.get('blip_caption'),
                            'similarity': float(dist),
                            'embedding_type': 'image'
                        })
                        if len(results) >= k_per_system:
                            break
            elif system == 'blip_text':
                results = self.search_early_fusion_blip_text(query, k_per_system)
            elif system == 'bm25':
                results = self.search_bm25(query, k_per_system)
            else:
                raise ValueError(f"Unknown system: {system}")
            
            # Normalize scores
            results = self._normalize_scores(results)
            system_results.append(results)
        
        # Combine scores
        combined_scores = defaultdict(lambda: {'score': 0.0, 'meta': None})
        
        for results, weight in zip(system_results, weights):
            for r in results:
                img_id = r['image_id']
                combined_scores[img_id]['score'] += weight * r['normalized_similarity']
                if combined_scores[img_id]['meta'] is None:
                    combined_scores[img_id]['meta'] = r
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:k]
        
        # Format output
        final_results = []
        for img_id, data in sorted_results:
            result = data['meta'].copy()
            result['combined_score'] = data['score']
            result['fusion_method'] = 'combsum'
            final_results.append(result)
        
        return final_results
    
    def late_fusion_max_pooling(self,
                                query: str,
                                systems: List[str],
                                k: int = 10,
                                k_per_system: int = 100) -> List[Dict]:
        """
        Late Fusion: Max-pooling (take maximum score for each image)
        
        Args:
            query: Text query
            systems: List of systems to combine
            k: Number of final results
            k_per_system: Number of results to retrieve from each system
            
        Returns:
            Fused results sorted by max score
        """
        logger.info(f"Late Fusion Max-Pooling with systems: {systems}")
        
        # Retrieve from each system
        system_results = []
        for system in systems:
            if system == 'image':
                query_embedding = self.encode_text(query)
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype('float32'), 
                    len(self.metadata)
                )
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    meta = self.metadata[idx]
                    if meta.get('embedding_type') == 'image':
                        results.append({
                            'image_id': meta.get('image_id'),
                            'image_path': meta.get('image_path'),
                            'caption': meta.get('caption'),
                            'blip_caption': meta.get('blip_caption'),
                            'similarity': float(dist),
                            'embedding_type': 'image'
                        })
                        if len(results) >= k_per_system:
                            break
            elif system == 'blip_text':
                results = self.search_early_fusion_blip_text(query, k_per_system)
            elif system == 'bm25':
                results = self.search_bm25(query, k_per_system)
            else:
                raise ValueError(f"Unknown system: {system}")
            
            # Normalize scores
            results = self._normalize_scores(results)
            system_results.append(results)
        
        # Take max score for each image
        max_scores = defaultdict(lambda: {'score': 0.0, 'meta': None})
        
        for results in system_results:
            for r in results:
                img_id = r['image_id']
                if r['normalized_similarity'] > max_scores[img_id]['score']:
                    max_scores[img_id]['score'] = r['normalized_similarity']
                    max_scores[img_id]['meta'] = r
        
        # Sort by max score
        sorted_results = sorted(
            max_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:k]
        
        # Format output
        final_results = []
        for img_id, data in sorted_results:
            result = data['meta'].copy()
            result['combined_score'] = data['score']
            result['fusion_method'] = 'max_pooling'
            final_results.append(result)
        
        return final_results
    
    def late_fusion_rrf(self,
                       query: str,
                       systems: List[str],
                       k: int = 10,
                       k_per_system: int = 100,
                       rrf_k: int = 60) -> List[Dict]:
        """
        Late Fusion: Reciprocal Rank Fusion (RRF)
        
        Args:
            query: Text query
            systems: List of systems to combine
            k: Number of final results
            k_per_system: Number of results to retrieve from each system
            rrf_k: RRF constant (typically 60)
            
        Returns:
            Fused results sorted by RRF score
        """
        logger.info(f"Late Fusion RRF with systems: {systems}")
        
        # Retrieve from each system
        system_results = []
        for system in systems:
            if system == 'image':
                query_embedding = self.encode_text(query)
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype('float32'), 
                    len(self.metadata)
                )
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    meta = self.metadata[idx]
                    if meta.get('embedding_type') == 'image':
                        results.append({
                            'image_id': meta.get('image_id'),
                            'image_path': meta.get('image_path'),
                            'caption': meta.get('caption'),
                            'blip_caption': meta.get('blip_caption'),
                            'similarity': float(dist),
                            'embedding_type': 'image'
                        })
                        if len(results) >= k_per_system:
                            break
            elif system == 'blip_text':
                results = self.search_early_fusion_blip_text(query, k_per_system)
            elif system == 'bm25':
                results = self.search_bm25(query, k_per_system)
            else:
                raise ValueError(f"Unknown system: {system}")
            
            system_results.append(results)
        
        # Calculate RRF scores
        rrf_scores = defaultdict(lambda: {'score': 0.0, 'meta': None})
        
        for results in system_results:
            for rank, r in enumerate(results, start=1):
                img_id = r['image_id']
                rrf_scores[img_id]['score'] += 1.0 / (rrf_k + rank)
                if rrf_scores[img_id]['meta'] is None:
                    rrf_scores[img_id]['meta'] = r
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:k]
        
        # Format output
        final_results = []
        for img_id, data in sorted_results:
            result = data['meta'].copy()
            result['combined_score'] = data['score']
            result['fusion_method'] = 'rrf'
            final_results.append(result)
        
        return final_results
    
    def late_fusion_borda(self,
                         query: str,
                         systems: List[str],
                         k: int = 10,
                         k_per_system: int = 100) -> List[Dict]:
        """
        Late Fusion: Borda Count
        
        Args:
            query: Text query
            systems: List of systems to combine
            k: Number of final results
            k_per_system: Number of results to retrieve from each system
            
        Returns:
            Fused results sorted by Borda count
        """
        logger.info(f"Late Fusion Borda with systems: {systems}")
        
        # Retrieve from each system
        system_results = []
        for system in systems:
            if system == 'image':
                query_embedding = self.encode_text(query)
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype('float32'), 
                    len(self.metadata)
                )
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    meta = self.metadata[idx]
                    if meta.get('embedding_type') == 'image':
                        results.append({
                            'image_id': meta.get('image_id'),
                            'image_path': meta.get('image_path'),
                            'caption': meta.get('caption'),
                            'blip_caption': meta.get('blip_caption'),
                            'similarity': float(dist),
                            'embedding_type': 'image'
                        })
                        if len(results) >= k_per_system:
                            break
            elif system == 'blip_text':
                results = self.search_early_fusion_blip_text(query, k_per_system)
            elif system == 'bm25':
                results = self.search_bm25(query, k_per_system)
            else:
                raise ValueError(f"Unknown system: {system}")
            
            system_results.append(results)
        
        # Calculate Borda points (n - rank)
        borda_scores = defaultdict(lambda: {'score': 0, 'meta': None})
        
        for results in system_results:
            n = len(results)
            for rank, r in enumerate(results):
                img_id = r['image_id']
                borda_scores[img_id]['score'] += (n - rank)
                if borda_scores[img_id]['meta'] is None:
                    borda_scores[img_id]['meta'] = r
        
        # Sort by Borda count
        sorted_results = sorted(
            borda_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:k]
        
        # Format output
        final_results = []
        for img_id, data in sorted_results:
            result = data['meta'].copy()
            result['combined_score'] = data['score']
            result['fusion_method'] = 'borda'
            final_results.append(result)
        
        return final_results
    
    def late_fusion_minimax_rank(self,
                                 query: str,
                                 systems: List[str],
                                 k: int = 10,
                                 k_per_system: int = 100) -> List[Dict]:
        """
        Late Fusion: Minimax-rank
        
        For each image in the merged list (sorted by similarity after normalization),
        find the position of its last view (2nd if 2 systems, 3rd if 3 systems).
        This is the T score. Then sort by T ascending (lower is better).
        
        Args:
            query: Text query
            systems: List of systems to combine
            k: Number of final results
            k_per_system: Number of results to retrieve from each system
            
        Returns:
            Fused results sorted by minimax-rank score
        """
        logger.info(f"Late Fusion Minimax-rank with systems: {systems}")
        
        # Retrieve from each system
        system_results = []
        for system in systems:
            if system == 'image':
                query_embedding = self.encode_text(query)
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype('float32'), 
                    len(self.metadata)
                )
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    meta = self.metadata[idx]
                    if meta.get('embedding_type') == 'image':
                        results.append({
                            'image_id': meta.get('image_id'),
                            'image_path': meta.get('image_path'),
                            'caption': meta.get('caption'),
                            'blip_caption': meta.get('blip_caption'),
                            'similarity': float(dist),
                            'embedding_type': 'image',
                            'system': system
                        })
                        if len(results) >= k_per_system:
                            break
            elif system == 'blip_text':
                results = self.search_early_fusion_blip_text(query, k_per_system)
                for r in results:
                    r['system'] = system
            elif system == 'bm25':
                results = self.search_bm25(query, k_per_system)
                for r in results:
                    r['system'] = system
            else:
                raise ValueError(f"Unknown system: {system}")
            
            # Normalize scores for this system
            results = self._normalize_scores(results)
            system_results.append(results)
        
        # Merge all results and sort by normalized similarity
        all_results = []
        for results in system_results:
            all_results.extend(results)
        
        # Sort by normalized similarity (descending)
        all_results.sort(key=lambda x: x['normalized_similarity'], reverse=True)
        
        # For each unique image, find positions of all its views
        image_positions = defaultdict(list)
        for pos, r in enumerate(all_results):
            img_id = r['image_id']
            image_positions[img_id].append({
                'position': pos,
                'system': r['system'],
                'result': r
            })
        
        # Calculate minimax score (position of last view)
        minimax_scores = []
        for img_id, positions in image_positions.items():
            # Sort positions to get the last view
            sorted_positions = sorted(positions, key=lambda x: x['position'])
            
            # The minimax score is the position of the last view
            # (2nd view if 2 systems, 3rd view if 3 systems, etc.)
            num_views = len(sorted_positions)
            
            # Only consider images that appear in the required number of systems
            if num_views >= len(systems):
                last_view_position = sorted_positions[len(systems) - 1]['position']
            elif num_views > 0:
                # If image doesn't appear in all systems, use last available view
                last_view_position = sorted_positions[-1]['position']
            else:
                continue
            
            minimax_scores.append({
                'image_id': img_id,
                'minimax_score': last_view_position,
                'num_views': num_views,
                'result': positions[0]['result']  # Use first result for metadata
            })
        
        # Sort by minimax score (ascending - lower is better)
        minimax_scores.sort(key=lambda x: x['minimax_score'])
        
        # Format output
        final_results = []
        for item in minimax_scores[:k]:
            result = item['result'].copy()
            result['combined_score'] = item['minimax_score']
            result['num_views'] = item['num_views']
            result['fusion_method'] = 'minimax_rank'
            final_results.append(result)
        
        return final_results
    
    # ==================== WRAPPER METHOD ====================
    
    def retrieve_from_image(self,
                           image_identifier: str,
                           fusion_method: str = 'combsum',
                           systems: Optional[List[str]] = None,
                           weights: Optional[List[float]] = None,
                           k: int = 10,
                           k_per_system: int = 100,
                           rrf_k: int = 60) -> Dict:
        """
        Retrieve similar images given an image identifier (wrapper method)
        
        Args:
            image_identifier: Image ID or filename from the dataset
            fusion_method: Fusion method to use
                          - 'early_average': Early fusion on average embeddings
                          - 'early_blip_text': Early fusion on BLIP text embeddings
                          - 'bm25': BM25 search on BLIP captions
                          - 'combsum': Late fusion with weighted CombSum
                          - 'max_pooling': Late fusion with max-pooling
                          - 'rrf': Late fusion with RRF
                          - 'borda': Late fusion with Borda count
                          - 'minimax_rank': Late fusion with minimax-rank
            systems: List of systems for late fusion ['image', 'blip_text', 'bm25']
                    (default: all three)
            weights: Weights for CombSum (default: equal weights)
            k: Number of results to return
            k_per_system: Number of results per system for late fusion
            rrf_k: RRF constant
            
        Returns:
            Dictionary with query info and results
        """
        # Get caption for the image
        caption = self.get_caption_for_image(image_identifier)
        
        if caption is None:
            logger.error(f"Could not find caption for image: {image_identifier}")
            return {
                'query_image': image_identifier,
                'query_caption': None,
                'fusion_method': fusion_method,
                'error': 'Caption not found'
            }
        
        logger.info(f"Retrieving for image: {image_identifier}")
        logger.info(f"Query caption: {caption}")
        logger.info(f"Fusion method: {fusion_method}")
        
        # Set default systems for late fusion
        if systems is None:
            systems = ['image', 'blip_text', 'bm25']
        
        # Execute retrieval based on method
        start_time = time.time()
        
        if fusion_method == 'early_average':
            results = self.search_early_fusion_average(caption, k)
        elif fusion_method == 'early_blip_text':
            results = self.search_early_fusion_blip_text(caption, k)
        elif fusion_method == 'bm25':
            results = self.search_bm25(caption, k)
        elif fusion_method == 'combsum':
            results = self.late_fusion_combsum(caption, systems, weights, k, k_per_system)
        elif fusion_method == 'max_pooling':
            results = self.late_fusion_max_pooling(caption, systems, k, k_per_system)
        elif fusion_method == 'rrf':
            results = self.late_fusion_rrf(caption, systems, k, k_per_system, rrf_k)
        elif fusion_method == 'borda':
            results = self.late_fusion_borda(caption, systems, k, k_per_system)
        elif fusion_method == 'minimax_rank':
            results = self.late_fusion_minimax_rank(caption, systems, k, k_per_system)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        retrieval_time = time.time() - start_time
        
        return {
            'query_image': image_identifier,
            'query_caption': caption,
            'fusion_method': fusion_method,
            'systems': systems if 'late' in fusion_method or fusion_method in ['combsum', 'max_pooling', 'rrf', 'borda', 'minimax_rank'] else None,
            'k': k,
            'results': results,
            'retrieval_time': retrieval_time,
            'num_results': len(results)
        }
    
    def display_results(self, retrieval_result: Dict, max_display: int = 10):
        """
        Display retrieval results
        
        Args:
            retrieval_result: Result from retrieve_from_image
            max_display: Maximum number of results to display
        """
        print("\n" + "="*80)
        print("RETRIEVAL RESULTS")
        print("="*80)
        print(f"Query Image: {retrieval_result['query_image']}")
        print(f"Query Caption: {retrieval_result['query_caption']}")
        print(f"Fusion Method: {retrieval_result['fusion_method']}")
        if retrieval_result.get('systems'):
            print(f"Systems: {retrieval_result['systems']}")
        print(f"Retrieved: {retrieval_result['num_results']} results in {retrieval_result['retrieval_time']:.3f}s")
        print("="*80)
        
        if 'error' in retrieval_result:
            print(f"ERROR: {retrieval_result['error']}")
            return
        
        results = retrieval_result['results'][:max_display]
        
        for idx, result in enumerate(results, 1):
            print(f"\n[{idx}] Image ID: {result['image_id']}")
            print(f"    Similarity: {result.get('similarity', 0):.4f}")
            if 'combined_score' in result:
                print(f"    Combined Score: {result['combined_score']:.4f}")
            if 'normalized_similarity' in result:
                print(f"    Normalized Similarity: {result['normalized_similarity']:.4f}")
            print(f"    Caption: {result['caption']}")
            print(f"    BLIP Caption: {result['blip_caption'][:100]}...")
            print(f"    Path: {result['image_path']}")
        
        print("\n" + "="*80)


def main():
    """Main function for testing the retriever"""
    parser = argparse.ArgumentParser(
        description='Double Embedding Retriever'
    )
    parser.add_argument(
        '--index',
        type=str,
        required=True,
        help='Path to FAISS index file'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata JSON file'
    )
    parser.add_argument(
        '--image-id',
        type=str,
        required=True,
        help='Image identifier to search for'
    )
    parser.add_argument(
        '--fusion-method',
        type=str,
        default='combsum',
        choices=['early_average', 'early_blip_text', 'bm25', 
                'combsum', 'max_pooling', 'rrf', 'borda', 'minimax_rank'],
        help='Fusion method to use'
    )
    parser.add_argument(
        '--systems',
        type=str,
        nargs='+',
        default=['image', 'blip_text', 'bm25'],
        help='Systems to combine for late fusion'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to return'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("DOUBLE EMBEDDING RETRIEVER")
    print("="*80)
    print(f"Index: {args.index}")
    print(f"Metadata: {args.metadata}")
    print(f"Image ID: {args.image_id}")
    print(f"Fusion Method: {args.fusion_method}")
    print(f"Systems: {args.systems}")
    print("="*80)
    
    # Create retriever
    retriever = DoubleEmbeddingRetriever(
        index_path=args.index,
        metadata_path=args.metadata
    )
    
    # Retrieve
    result = retriever.retrieve_from_image(
        image_identifier=args.image_id,
        fusion_method=args.fusion_method,
        systems=args.systems,
        k=args.k
    )
    
    # Display results
    retriever.display_results(result)
    
    return 0


if __name__ == "__main__":
    exit(main())
