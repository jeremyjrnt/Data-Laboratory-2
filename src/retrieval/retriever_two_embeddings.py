"""
Two Embeddings Retrieval System for Flickr Dataset
Supports multiple retrieval systems and fusion methods
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi


class TwoEmbeddingsRetriever:
    """
    Retrieval system supporting multiple approaches:
    - Image embedding (classic VectorDB)
    - Text embedding (BLIP caption VectorDB)
    - Average embedding (VectorDB)
    - BM25 on text corpus
    
    Fusion methods:
    - CombSUM (normalized Min-Max)
    - RRF (Reciprocal Rank Fusion)
    - Borda Count
    """
    
    def __init__(self, dataset_name='Flickr', k=100):
        """
        Initialize the retriever
        
        Args:
            dataset_name: Name of the dataset (default: Flickr)
            k: Number of results to retrieve (default: 100)
        """
        self.dataset_name = dataset_name
        self.k = k
        self.model_name = Config.HF_MODEL_CLIP_LARGE
        self.dimension = 768
        
        # CLIP model and processor
        self.model = None
        self.processor = None
        self.device = None
        
        # Paths
        self.base_path = Config.PROJECT_ROOT
        self.data_path = Config.get_dataset_dir(dataset_name)
        self.images_path = Config.get_images_dir(dataset_name)
        self.vectordb_path = Config.VECTORDB_DIR
        
        # Metadata path for true captions
        self.metadata_path = self.data_path / f"{dataset_name.lower()}_metadata.json"
        
        # VectorDB paths
        self.classic_index_path = self.vectordb_path / f"{dataset_name}_VectorDB.index"
        self.classic_metadata_path = self.vectordb_path / f"{dataset_name}_VectorDB_metadata.json"
        
        self.blip_index_path = self.vectordb_path / f"{dataset_name}_blip_caption_VectorDB.index"
        self.blip_metadata_path = self.vectordb_path / f"{dataset_name}_blip_caption_VectorDB_metadata.json"
        
        self.avg_index_path = self.vectordb_path / f"{dataset_name}_average_VectorDB.index"
        self.avg_metadata_path = self.vectordb_path / f"{dataset_name}_average_VectorDB_metadata.json"
        
        # BM25 corpus path
        self.corpus_path = self.data_path / f"{dataset_name.lower()}_corpus.json"
        
        # Loaded indexes and metadata
        self.classic_index = None
        self.classic_metadata = []
        
        self.blip_index = None
        self.blip_metadata = []
        
        self.avg_index = None
        self.avg_metadata = []
        
        # BM25
        self.bm25 = None
        self.corpus_dict = {}
        self.corpus_filenames = []
        
        # True captions metadata
        self.true_metadata = []
        
    def _load_clip_model(self):
        """Load CLIP model and processor on GPU"""
        if self.model is not None:
            return  # Already loaded
        
        print("🔄 Loading CLIP model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
        
    def load_classic_vectordb(self):
        """Load classic image embedding VectorDB"""
        print(f"🔄 Loading classic VectorDB from {self.classic_index_path}...")
        
        if not self.classic_index_path.exists():
            raise FileNotFoundError(f"Classic index not found: {self.classic_index_path}")
        
        self.classic_index = faiss.read_index(str(self.classic_index_path))
        
        with open(self.classic_metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.classic_metadata = data['metadata']
        
        print(f"✓ Loaded classic VectorDB with {self.classic_index.ntotal} vectors")
        
    def load_blip_vectordb(self):
        """Load BLIP caption text embedding VectorDB"""
        print(f"🔄 Loading BLIP caption VectorDB from {self.blip_index_path}...")
        
        if not self.blip_index_path.exists():
            raise FileNotFoundError(f"BLIP index not found: {self.blip_index_path}")
        
        self.blip_index = faiss.read_index(str(self.blip_index_path))
        
        with open(self.blip_metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.blip_metadata = data['metadata']
        
        print(f"✓ Loaded BLIP VectorDB with {self.blip_index.ntotal} vectors")
        
    def load_average_vectordb(self):
        """Load average (image + text) embedding VectorDB"""
        print(f"🔄 Loading average VectorDB from {self.avg_index_path}...")
        
        if not self.avg_index_path.exists():
            raise FileNotFoundError(f"Average index not found: {self.avg_index_path}")
        
        self.avg_index = faiss.read_index(str(self.avg_index_path))
        
        with open(self.avg_metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.avg_metadata = data['metadata']
        
        print(f"✓ Loaded average VectorDB with {self.avg_index.ntotal} vectors")
        
    def load_true_metadata(self):
        """Load true captions from flickr_metadata.json"""
        if len(self.true_metadata) > 0:
            return  # Already loaded
        
        print(f"🔄 Loading true captions from {self.metadata_path}...")
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'images' in data:
            self.true_metadata = data['images']
        elif isinstance(data, list):
            self.true_metadata = data
        else:
            raise ValueError("Unexpected metadata format")
        
        print(f"✓ Loaded {len(self.true_metadata)} true captions")
    
    def load_bm25(self):
        """Load BM25 corpus"""
        print(f"🔄 Loading BM25 corpus from {self.corpus_path}...")
        
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            self.corpus_dict = json.load(f)
        
        # Prepare corpus for BM25
        self.corpus_filenames = list(self.corpus_dict.keys())
        corpus_texts = [self.corpus_dict[fn].lower().split() for fn in self.corpus_filenames]
        
        self.bm25 = BM25Okapi(corpus_texts)
        
        print(f"✓ Loaded BM25 corpus with {len(self.corpus_filenames)} documents")
        
    def get_baseline_rank(self, filename: str) -> Optional[int]:
        """Get baseline rank for a filename from flickr_metadata.json"""
        self.load_true_metadata()
        
        for item in self.true_metadata:
            item_filename = item.get('filename') or item.get('file_name') or item.get('image')
            if item_filename == filename:
                return item.get('baseline_rank')
        return None
    
    def get_true_caption(self, filename: str) -> Optional[str]:
        """Get true caption for a filename from flickr_metadata.json"""
        self.load_true_metadata()
        
        for item in self.true_metadata:
            item_filename = item.get('filename') or item.get('file_name') or item.get('image')
            if item_filename == filename:
                return item.get('caption', '')
        return None
        
    def encode_image(self, image_path: Path) -> np.ndarray:
        """Encode image to CLIP embedding"""
        self._load_clip_model()
        
        image = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
            
            embedding = image_features.cpu().numpy().astype('float32')[0]
        
        return embedding
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to CLIP embedding"""
        self._load_clip_model()
        
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_features = self.model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            text_features = F.normalize(text_features, p=2, dim=1)
            
            embedding = text_features.cpu().numpy().astype('float32')[0]
        
        return embedding
        
    def search_classic(self, query_embedding: np.ndarray, k: int = None) -> List[Dict]:
        """Search using classic image VectorDB"""
        if k is None:
            k = self.k
        
        distances, indices = self.classic_index.search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for idx, (dist, i) in enumerate(zip(distances[0], indices[0])):
            if i < len(self.classic_metadata):
                item = self.classic_metadata[i]
                filename = item.get('filename') or item.get('file_name') or item.get('image')
                results.append({
                    'rank': idx + 1,
                    'filename': filename,
                    'score': float(dist),
                    'metadata': item
                })
        
        return results
        
    def search_blip(self, query_embedding: np.ndarray, k: int = None) -> List[Dict]:
        """Search using BLIP caption VectorDB"""
        if k is None:
            k = self.k
        
        distances, indices = self.blip_index.search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for idx, (dist, i) in enumerate(zip(distances[0], indices[0])):
            if i < len(self.blip_metadata):
                item = self.blip_metadata[i]
                filename = item.get('filename') or item.get('file_name') or item.get('image')
                results.append({
                    'rank': idx + 1,
                    'filename': filename,
                    'score': float(dist),
                    'metadata': item
                })
        
        return results
        
    def search_average(self, query_embedding: np.ndarray, k: int = None) -> List[Dict]:
        """Search using average VectorDB"""
        if k is None:
            k = self.k
        
        distances, indices = self.avg_index.search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for idx, (dist, i) in enumerate(zip(distances[0], indices[0])):
            if i < len(self.avg_metadata):
                item = self.avg_metadata[i]
                filename = item.get('filename') or item.get('file_name') or item.get('image')
                results.append({
                    'rank': idx + 1,
                    'filename': filename,
                    'score': float(dist),
                    'metadata': item
                })
        
        return results
        
    def search_bm25(self, query_text: str, k: int = None) -> List[Dict]:
        """Search using BM25"""
        if k is None:
            k = self.k
        
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get ALL indices sorted by score (exhaustive search)
        all_indices = np.argsort(scores)[::-1]
        
        # Return top k results
        results = []
        for idx, i in enumerate(all_indices[:k]):
            filename = self.corpus_filenames[i]
            results.append({
                'rank': idx + 1,
                'filename': filename,
                'score': float(scores[i]),
                'text': self.corpus_dict[filename]
            })
        
        return results
        
    def normalize_scores_minmax(self, results: List[Dict]) -> List[Dict]:
        """Normalize scores using Min-Max normalization"""
        scores = [r['score'] for r in results]
        
        if len(scores) == 0:
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for r in results:
                r['normalized_score'] = 1.0
        else:
            for r in results:
                r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)
        
        return results
        
    def fuse_combsum(self, results_list: List[List[Dict]], return_all: bool = False) -> List[Dict]:
        """
        Fuse multiple result lists using CombSUM with Min-Max normalization
        
        Args:
            results_list: List of result lists to fuse
            return_all: If True, return all results instead of top k
        """
        if not return_all:
            print("🔄 Applying CombSUM fusion...")
        
        # Normalize each result list
        normalized_results = []
        for results in results_list:
            normalized_results.append(self.normalize_scores_minmax(results))
        
        # Aggregate scores by filename
        score_dict = {}
        
        for results in normalized_results:
            for item in results:
                filename = item['filename']
                score = item['normalized_score']
                
                if filename not in score_dict:
                    score_dict[filename] = {
                        'filename': filename,
                        'score': 0.0,
                        'metadata': item.get('metadata', {})
                    }
                
                score_dict[filename]['score'] += score
        
        # Sort by combined score
        fused_results = sorted(score_dict.values(), key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for idx, item in enumerate(fused_results):
            item['rank'] = idx + 1
        
        return fused_results if return_all else fused_results[:self.k]
        
    def fuse_rrf(self, results_list: List[List[Dict]], k_rrf: int = 60, return_all: bool = False) -> List[Dict]:
        """
        Fuse multiple result lists using Reciprocal Rank Fusion (RRF)
        
        Args:
            results_list: List of result lists to fuse
            k_rrf: k parameter for RRF
            return_all: If True, return all results instead of top k
        """
        if not return_all:
            print("🔄 Applying RRF fusion...")
        
        # Aggregate RRF scores by filename
        score_dict = {}
        
        for results in results_list:
            for item in results:
                filename = item['filename']
                rank = item['rank']
                rrf_score = 1.0 / (k_rrf + rank)
                
                if filename not in score_dict:
                    score_dict[filename] = {
                        'filename': filename,
                        'score': 0.0,
                        'metadata': item.get('metadata', {})
                    }
                
                score_dict[filename]['score'] += rrf_score
        
        # Sort by RRF score
        fused_results = sorted(score_dict.values(), key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for idx, item in enumerate(fused_results):
            item['rank'] = idx + 1
        
        return fused_results if return_all else fused_results[:self.k]
        
    def fuse_borda(self, results_list: List[List[Dict]], return_all: bool = False) -> List[Dict]:
        """
        Fuse multiple result lists using Borda Count
        
        Args:
            results_list: List of result lists to fuse
            return_all: If True, return all results instead of top k
        """
        if not return_all:
            print("🔄 Applying Borda Count fusion...")
        
        # Get maximum rank (for Borda points calculation)
        max_rank = max(len(results) for results in results_list)
        
        # Aggregate Borda points by filename
        score_dict = {}
        
        for results in results_list:
            for item in results:
                filename = item['filename']
                rank = item['rank']
                borda_points = max_rank - rank
                
                if filename not in score_dict:
                    score_dict[filename] = {
                        'filename': filename,
                        'score': 0,
                        'metadata': item.get('metadata', {})
                    }
                
                score_dict[filename]['score'] += borda_points
        
        # Sort by Borda points
        fused_results = sorted(score_dict.values(), key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for idx, item in enumerate(fused_results):
            item['rank'] = idx + 1
        
        return fused_results if return_all else fused_results[:self.k]
        
    def retrieve(
        self,
        filename: str,
        systems: List[str],
        fusion_method: Optional[str] = None,
        k_rrf: int = 60
    ) -> Dict:
        """
        Main retrieval function
        
        Args:
            filename: Image filename to use as query
            systems: List of systems to use. Options: 'classic', 'blip', 'average', 'bm25'
            fusion_method: Fusion method if multiple systems. Options: 'combsum', 'rrf', 'borda'
            k_rrf: k parameter for RRF (default: 60)
        
        Returns:
            Dictionary with results and metadata
        """
        print(f"\n{'='*70}")
        print(f"Two Embeddings Retrieval for {filename}")
        print('='*70)
        
        # Get baseline rank
        baseline_rank = self.get_baseline_rank(filename)
        if baseline_rank:
            print(f"📊 Baseline Rank: {baseline_rank}")
        else:
            print("⚠️  Baseline rank not found")
        
        # Validate systems
        valid_systems = ['classic', 'blip', 'average', 'bm25']
        for system in systems:
            if system not in valid_systems:
                raise ValueError(f"Invalid system: {system}. Valid options: {valid_systems}")
        
        # Load required resources
        print(f"\n🔄 Loading resources for systems: {', '.join(systems)}")
        
        results_list = []
        
        # Prepare query
        image_path = self.images_path / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get true caption from flickr_metadata.json
        true_caption = self.get_true_caption(filename)
        
        if not true_caption:
            raise ValueError(f"True caption not found for {filename}")
        
        print(f"📝 True Caption: {true_caption[:100]}...")
        
        # Execute searches for each system
        print(f"\n🔍 Executing searches...")
        
        if 'classic' in systems:
            self.load_classic_vectordb()
            print("  ⚡ Searching classic image VectorDB...")
            query_emb = self.encode_text(true_caption)
            results = self.search_classic(query_emb)
            results_list.append(results)
            print(f"     ✓ Found {len(results)} results")
        
        if 'blip' in systems:
            self.load_blip_vectordb()
            print("  ⚡ Searching BLIP caption VectorDB...")
            query_emb = self.encode_text(true_caption)
            results = self.search_blip(query_emb)
            results_list.append(results)
            print(f"     ✓ Found {len(results)} results")
        
        if 'average' in systems:
            self.load_average_vectordb()
            print("  ⚡ Searching average VectorDB...")
            query_emb = self.encode_text(true_caption)
            results = self.search_average(query_emb)
            results_list.append(results)
            print(f"     ✓ Found {len(results)} results")
        
        if 'bm25' in systems:
            self.load_bm25()
            print("  ⚡ Searching BM25...")
            results = self.search_bm25(true_caption)
            results_list.append(results)
            print(f"     ✓ Found {len(results)} results")
        
        # Fuse results if multiple systems
        if len(systems) > 1:
            if not fusion_method:
                raise ValueError("Fusion method required when using multiple systems. Options: 'combsum', 'rrf', 'borda'")
            
            print(f"\n🔀 Fusing results with {fusion_method.upper()}...")
            
            if fusion_method == 'combsum':
                final_results = self.fuse_combsum(results_list)
            elif fusion_method == 'rrf':
                final_results = self.fuse_rrf(results_list, k_rrf=k_rrf)
            elif fusion_method == 'borda':
                final_results = self.fuse_borda(results_list)
            else:
                raise ValueError(f"Invalid fusion method: {fusion_method}")
        else:
            final_results = results_list[0]
        
        # Find new rank with exhaustive search if not in top k
        new_rank = None
        for item in final_results:
            if item['filename'] == filename:
                new_rank = item['rank']
                break
        
        # If not found in top k, perform exhaustive search to find exact rank
        if new_rank is None:
            print(f"\n⚠️  Target not in top {self.k} results, performing exhaustive search...")
            
            if len(systems) == 1:
                # Single system exhaustive search
                system = systems[0]
                
                if system == 'bm25':
                    # BM25: search through all scores
                    tokenized_query = true_caption.lower().split()
                    all_scores = self.bm25.get_scores(tokenized_query)
                    all_indices = np.argsort(all_scores)[::-1]
                    
                    for rank, idx in enumerate(all_indices, 1):
                        if self.corpus_filenames[idx] == filename:
                            new_rank = rank
                            break
                else:
                    # Vector DB: search with larger k incrementally
                    max_search = self.classic_index.ntotal if system == 'classic' else \
                                 self.blip_index.ntotal if system == 'blip' else \
                                 self.avg_index.ntotal
                    
                    query_emb = self.encode_text(true_caption)
                    
                    if system == 'classic':
                        distances, indices = self.classic_index.search(query_emb.reshape(1, -1), max_search)
                        metadata = self.classic_metadata
                    elif system == 'blip':
                        distances, indices = self.blip_index.search(query_emb.reshape(1, -1), max_search)
                        metadata = self.blip_metadata
                    elif system == 'average':
                        distances, indices = self.avg_index.search(query_emb.reshape(1, -1), max_search)
                        metadata = self.avg_metadata
                    
                    for rank, idx in enumerate(indices[0], 1):
                        if idx < len(metadata):
                            item = metadata[idx]
                            item_filename = item.get('filename') or item.get('file_name') or item.get('image')
                            if item_filename == filename:
                                new_rank = rank
                                break
            else:
                # Multiple systems: need to recompute full fusion
                print("   🔄 Recomputing fusion with all results...")
                
                # Get all results for each system
                all_results_list = []
                
                if 'classic' in systems:
                    query_emb = self.encode_text(true_caption)
                    distances, indices = self.classic_index.search(query_emb.reshape(1, -1), self.classic_index.ntotal)
                    results = []
                    for idx, (dist, i) in enumerate(zip(distances[0], indices[0])):
                        if i < len(self.classic_metadata):
                            item = self.classic_metadata[i]
                            item_filename = item.get('filename') or item.get('file_name') or item.get('image')
                            results.append({'rank': idx + 1, 'filename': item_filename, 'score': float(dist), 'metadata': item})
                    all_results_list.append(results)
                
                if 'blip' in systems:
                    query_emb = self.encode_text(true_caption)
                    distances, indices = self.blip_index.search(query_emb.reshape(1, -1), self.blip_index.ntotal)
                    results = []
                    for idx, (dist, i) in enumerate(zip(distances[0], indices[0])):
                        if i < len(self.blip_metadata):
                            item = self.blip_metadata[i]
                            item_filename = item.get('filename') or item.get('file_name') or item.get('image')
                            results.append({'rank': idx + 1, 'filename': item_filename, 'score': float(dist), 'metadata': item})
                    all_results_list.append(results)
                
                if 'average' in systems:
                    query_emb = self.encode_text(true_caption)
                    distances, indices = self.avg_index.search(query_emb.reshape(1, -1), self.avg_index.ntotal)
                    results = []
                    for idx, (dist, i) in enumerate(zip(distances[0], indices[0])):
                        if i < len(self.avg_metadata):
                            item = self.avg_metadata[i]
                            item_filename = item.get('filename') or item.get('file_name') or item.get('image')
                            results.append({'rank': idx + 1, 'filename': item_filename, 'score': float(dist), 'metadata': item})
                    all_results_list.append(results)
                
                if 'bm25' in systems:
                    tokenized_query = true_caption.lower().split()
                    all_scores = self.bm25.get_scores(tokenized_query)
                    all_indices = np.argsort(all_scores)[::-1]
                    results = []
                    for idx, i in enumerate(all_indices):
                        item_filename = self.corpus_filenames[i]
                        results.append({'rank': idx + 1, 'filename': item_filename, 'score': float(all_scores[i])})
                    all_results_list.append(results)
                
                # Fuse all results (return all)
                if fusion_method == 'combsum':
                    all_fused = self.fuse_combsum(all_results_list, return_all=True)
                elif fusion_method == 'rrf':
                    all_fused = self.fuse_rrf(all_results_list, k_rrf=k_rrf, return_all=True)
                elif fusion_method == 'borda':
                    all_fused = self.fuse_borda(all_results_list, return_all=True)
                
                # Find rank in full results
                for item in all_fused:
                    if item['filename'] == filename:
                        new_rank = item['rank']
                        break
            
            if new_rank:
                print(f"   ✓ Found exact rank: {new_rank}")
            else:
                print(f"   ✗ Target image not found in corpus")
        
        # Display progression
        print(f"\n{'='*70}")
        print("📈 RETRIEVAL RESULTS")
        print('='*70)
        
        if baseline_rank:
            print(f"📊 Baseline Rank: {baseline_rank}")
        
        if new_rank:
            print(f"🎯 New Rank: {new_rank}")
            
            if baseline_rank:
                if new_rank < baseline_rank:
                    improvement = baseline_rank - new_rank
                    print(f"✅ Rank Improvement: {baseline_rank} → {new_rank} (↑{improvement} positions)")
                elif new_rank > baseline_rank:
                    degradation = new_rank - baseline_rank
                    print(f"❌ Rank Degradation: {baseline_rank} → {new_rank} (↓{degradation} positions)")
                else:
                    print(f"🔄 Rank Unchanged: {baseline_rank} → {new_rank}")
        else:
            print("⚠️  Target image not found in results")
        
        print('='*70)
        
        # Show top 10 results
        print(f"\n🏆 Top 10 Results:")
        for i, item in enumerate(final_results[:10]):
            marker = "🎯" if item['filename'] == filename else "  "
            print(f"{marker} {i+1}. {item['filename'][:50]:<50} (score: {item['score']:.4f})")
        
        return {
            'filename': filename,
            'baseline_rank': baseline_rank,
            'new_rank': new_rank,
            'systems': systems,
            'fusion_method': fusion_method,
            'results': final_results[:10],
            'total_results': len(final_results)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Two Embeddings Retrieval System for Flickr'
    )
    parser.add_argument(
        '--filename',
        type=str,
        required=True,
        help='Image filename to use as query'
    )
    parser.add_argument(
        '--systems',
        type=str,
        nargs='+',
        required=True,
        choices=['classic', 'blip', 'average', 'bm25'],
        help='Retrieval systems to use (space-separated)'
    )
    parser.add_argument(
        '--fusion',
        type=str,
        choices=['combsum', 'rrf', 'borda'],
        help='Fusion method (required if multiple systems)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=100,
        help='Number of results to retrieve (default: 100)'
    )
    parser.add_argument(
        '--k_rrf',
        type=int,
        default=60,
        help='k parameter for RRF fusion (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Validate fusion method
    if len(args.systems) > 1 and not args.fusion:
        parser.error("--fusion is required when using multiple systems")
    
    retriever = TwoEmbeddingsRetriever(dataset_name='Flickr', k=args.k)
    
    result = retriever.retrieve(
        filename=args.filename,
        systems=args.systems,
        fusion_method=args.fusion,
        k_rrf=args.k_rrf
    )
    
    print(f"\n✓ Retrieval completed successfully!")


if __name__ == "__main__":
    main()
