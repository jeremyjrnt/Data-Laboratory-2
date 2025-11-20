#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVF Hybrid Retrieval with LLM Cluster Summaries
Combines embedding-based ranking with BM25 on cluster summaries using various fusion methods

python src/retrieval/retriever_ivf_llm_desc.py --filename 000000000077.jpg --llm-name gemma3_4b --fusion-method rrf --k-clusters 10
"""

import json
import sys
import argparse
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25Plus, BM25L
from config.config import Config


class HybridIVFRetriever:
    """
    Hybrid retrieval combining:
    - Embedding distance (CLIP) between query and centroids
    - BM25 on LLM-generated cluster summaries
    """
    
    def __init__(
        self,
        dataset: str = "COCO",
        llm_name: str = "gemma3_4b",
        fusion_method: str = "combsum",
        device: str = None,
        bm25_variant: str = "okapi",
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        bm25_delta: float = 1.0
    ):
        """
        Args:
            dataset: Dataset name (e.g., "COCO")
            llm_name: LLM model name for cluster summaries (e.g., "gemma3_4b", "mistral_7b")
            fusion_method: Fusion method (combsum, borda, max_pooling, rrf)
            device: Device for model inference (cuda/cpu)
            bm25_variant: BM25 variant (okapi, plus, l)
            bm25_k1: BM25 k1 parameter (term frequency saturation)
            bm25_b: BM25 b parameter (document length normalization)
            bm25_delta: BM25+ delta parameter (lower bound for term frequency)
        """
        self.dataset = dataset
        self.llm_name = llm_name
        self.fusion_method = fusion_method.lower()
        self.bm25_variant = bm25_variant.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.vectordb_dir = Config.VECTORDB_DIR
        self.data_dir = Config.get_dataset_dir(dataset)
        
        # Files
        self.index_path = self.vectordb_dir / f"{dataset}_IVF_KMeans.index"
        self.metadata_path = self.vectordb_dir / f"{dataset}_VectorDB_metadata.json"
        self.llm_clusters_path = self.vectordb_dir / f"{dataset}_IVF_CC_{llm_name}.json"
        self.captions_path = self.data_dir / f"{dataset.lower()}_metadata.json"
        self.cluster_positions_path = self.data_dir / "cluster_positions.json"
        
        # Validate BM25 variant
        valid_bm25_variants = ["okapi", "plus", "l", "short"]
        if self.bm25_variant not in valid_bm25_variants:
            raise ValueError(f"Invalid bm25_variant: {bm25_variant}. Must be one of {valid_bm25_variants}")
        
        # Validate fusion method
        valid_methods = ["combsum", "borda", "rrf"]
        if self.fusion_method not in valid_methods:
            raise ValueError(f"Invalid fusion_method: {fusion_method}. Must be one of {valid_methods}")
        
        # Fusion parameters (optimized for the 3 core methods)
        
        # BM25 parameters (optimized for information retrieval)
        self.bm25_k1 = bm25_k1  # Term frequency saturation parameter (recommended: 1.2, range: 1.2-2.0)
        self.bm25_b = bm25_b  # Document length normalization (recommended: 0.75, range: 0-1)
        self.bm25_delta = bm25_delta  # BM25+ delta parameter (recommended: 1.0)        # Load components
        self._load_index()
        self._load_metadata()
        self._load_llm_clusters()
        self._load_clip_model()
        self._build_bm25_index()
        self._load_captions()
        self._build_image_to_cluster_mapping()
        
        print(f"✅ HybridIVFRetriever initialized with BM25 variant: {self.bm25_variant}, fusion method: {self.fusion_method}")
    
    def _load_index(self):
        """Load FAISS IVF index"""
        print(f"\n📂 Loading IVF index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))
        self.nlist = self.index.nlist
        print(f"✅ Loaded IVF index: {self.index.ntotal} vectors, {self.nlist} clusters")
    
    def _load_metadata(self):
        """Load vector database metadata"""
        print(f"\n📂 Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
            self.metadata = meta_json.get("metadata", [])
        
        # Create filename to index mapping
        self.filename_to_idx = {meta["filename"]: idx for idx, meta in enumerate(self.metadata)}
        print(f"✅ Loaded {len(self.metadata)} metadata entries")
    
    def _load_llm_clusters(self):
        """Load LLM cluster summaries"""
        print(f"\n📂 Loading LLM cluster summaries from {self.llm_clusters_path}...")
        with open(self.llm_clusters_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.cluster_summaries = data.get("cluster_summaries", [])
        print(f"✅ Loaded {len(self.cluster_summaries)} cluster summaries")
    
    def _load_clip_model(self):
        """Load CLIP model for text encoding"""
        print(f"\n🤖 Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(Config.HF_MODEL_CLIP_LARGE).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(Config.HF_MODEL_CLIP_LARGE)
        self.clip_model.eval()
        print(f"✅ CLIP model loaded on {self.device}")
    
    def _build_bm25_index(self):
        """Build BM25 index on cluster summaries
        
        Creates a BM25 index over LLM-generated cluster descriptions to enable
        textual retrieval. This allows queries to be matched against semantic
        cluster summaries rather than just embedding distances, providing a
        hybrid retrieval approach combining vector search with text matching.
        
        The index is cached to disk to avoid rebuilding on every run.
        """
        # Define cache path in data/COCO directory
        bm25_cache_path = self.data_dir / f"bm25_cache_{self.llm_name}_{self.bm25_variant}_k1_{self.bm25_k1}_b_{self.bm25_b}_delta_{self.bm25_delta}.pkl"
        
        # Check if cached BM25 index already exists
        print(f"\n🔍 Checking for existing BM25 cache at: {bm25_cache_path}")
        
        if bm25_cache_path.exists():
            print(f"✅ Found existing BM25 cache file")
            print(f"📂 Loading cached BM25 index from {bm25_cache_path}...")
            try:
                with open(bm25_cache_path, "rb") as f:
                    cache = pickle.load(f)
                    self.bm25 = cache["bm25"]
                    self.cluster_id_to_summary_idx = cache["cluster_id_to_summary_idx"]
                print(f"✅ BM25 index successfully loaded from cache (variant: {self.bm25_variant})")
                return
            except Exception as e:
                print(f"⚠️  Failed to load cached BM25 index: {e}")
                print(f"   Rebuilding BM25 index...")
        else:
            print(f"❌ No existing BM25 cache found at expected location")
        
        print(f"\n🔍 Building new BM25 index on cluster summaries (variant: {self.bm25_variant})...")
        
        # Extract summaries from each cluster
        self.cluster_id_to_summary_idx = {}
        corpus = []
        
        for idx, cluster_info in enumerate(self.cluster_summaries):
            cluster_id = cluster_info.get("cluster_id")
            summary_list = cluster_info.get("summary", [])
            
            # Join all summaries for this cluster into one document
            summary_text = " ".join(summary_list) if isinstance(summary_list, list) else str(summary_list)
            
            # Tokenize (simple whitespace tokenization)
            tokenized_summary = summary_text.lower().split()
            corpus.append(tokenized_summary)
            
            self.cluster_id_to_summary_idx[cluster_id] = idx
        
        # Build BM25 index with selected variant
        if self.bm25_variant == "plus":
            self.bm25 = BM25Plus(corpus, k1=self.bm25_k1, b=self.bm25_b, delta=self.bm25_delta)
            print(f"✅ BM25+ index built on {len(corpus)} cluster summaries (k1={self.bm25_k1}, b={self.bm25_b}, delta={self.bm25_delta})")
        elif self.bm25_variant == "l":
            self.bm25 = BM25L(corpus, k1=self.bm25_k1, b=self.bm25_b, delta=self.bm25_delta)
            print(f"✅ BM25L index built on {len(corpus)} cluster summaries (k1={self.bm25_k1}, b={self.bm25_b}, delta={self.bm25_delta})")
        elif self.bm25_variant == "short":
            # BM25 optimized for short captions: lower k1 (0.9-1.2) and lower b (0.3-0.5)
            k1_short = 1.0  # Default for short texts (can be overridden by bm25_k1 parameter)
            b_short = 0.4   # Lower b reduces length normalization penalty for short texts
            
            # Use user-provided k1 if in the recommended range, otherwise use default
            if 0.9 <= self.bm25_k1 <= 1.2:
                k1_short = self.bm25_k1
            
            # Use user-provided b if reasonable, otherwise use default
            if 0.3 <= self.bm25_b <= 0.6:
                b_short = self.bm25_b
            
            self.bm25 = BM25Okapi(corpus, k1=k1_short, b=b_short)
            print(f"✅ BM25-Short index built on {len(corpus)} cluster summaries (k1={k1_short}, b={b_short}) - optimized for short captions")
        else:  # okapi (default)
            self.bm25 = BM25Okapi(corpus, k1=self.bm25_k1, b=self.bm25_b)
            print(f"✅ BM25Okapi index built on {len(corpus)} cluster summaries (k1={self.bm25_k1}, b={self.bm25_b})")
        
        # Save to cache
        try:
            print(f"💾 Saving BM25 index to cache: {bm25_cache_path}...")
            with open(bm25_cache_path, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "cluster_id_to_summary_idx": self.cluster_id_to_summary_idx
                }, f)
            print(f"✅ BM25 index saved to cache")
        except Exception as e:
            print(f"⚠️  Failed to save BM25 index to cache: {e}")
    
    def _load_captions(self):
        """Load image captions"""
        print(f"\n📂 Loading captions from {self.captions_path}...")
        with open(self.captions_path, "r", encoding="utf-8") as f:
            captions_raw = json.load(f)
        
        # Extract images list from COCO metadata format
        if isinstance(captions_raw, dict) and 'images' in captions_raw:
            self.captions_data = captions_raw['images']
        else:
            self.captions_data = captions_raw
        
        # Create filename to caption mapping
        self.filename_to_caption = {}
        for item in self.captions_data:
            filename = item.get("filename") or item.get("file_name")
            caption = item.get("caption") or item.get("captions") or item.get("caption_str")
            
            if filename and caption:
                if isinstance(caption, list) and len(caption) > 0:
                    caption = caption[0]
                self.filename_to_caption[filename] = caption
        
        print(f"✅ Loaded {len(self.filename_to_caption)} captions")
    
    def _build_image_to_cluster_mapping(self):
        """Build mapping from image index to cluster ID"""
        print(f"\n🔗 Building image-to-cluster mapping...")
        
        self.image_to_cluster = {}
        invlists = faiss.extract_index_ivf(self.index).invlists
        
        for cluster_id in tqdm(range(self.nlist), desc="Scanning clusters"):
            list_size = invlists.list_size(cluster_id)
            
            if list_size > 0:
                ids = faiss.rev_swig_ptr(invlists.get_ids(cluster_id), list_size)
                for img_idx in ids:
                    self.image_to_cluster[int(img_idx)] = cluster_id
        
        print(f"✅ Mapped {len(self.image_to_cluster)} images to clusters")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text with CLIP"""
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
            query_vector = text_features.cpu().numpy().astype('float32')
        
        return query_vector
    
    def _get_embedding_ranking(self, query_vector: np.ndarray, k: int) -> Tuple[List[int], List[float], List[int], List[float]]:
        """Get cluster ranking based on embedding distance"""
        # Search quantizer (centroids) - get all clusters to find real position
        D, I = self.index.quantizer.search(query_vector, self.nlist)
        all_cluster_ids = I[0].tolist()
        all_distances = D[0].tolist()
        
        # Return top-k and all
        cluster_ids = all_cluster_ids[:k]
        distances = all_distances[:k]
        
        return cluster_ids, distances, all_cluster_ids, all_distances
    
    def _get_bm25_ranking(self, query_text: str, k: int) -> Tuple[List[int], List[float]]:
        """Get cluster ranking based on BM25 scores on summaries"""
        # Tokenize query
        tokenized_query = query_text.lower().split()
        
        # Get BM25 scores for all clusters
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get all clusters sorted by score
        all_indices = np.argsort(bm25_scores)[::-1]
        
        # Map to cluster IDs for all
        all_cluster_ids = []
        for idx in all_indices:
            for cluster_id, summary_idx in self.cluster_id_to_summary_idx.items():
                if summary_idx == idx:
                    all_cluster_ids.append(cluster_id)
                    break
        
        # Get top-k
        cluster_ids = all_cluster_ids[:k]
        top_indices = all_indices[:k]
        scores = [bm25_scores[idx] for idx in top_indices]
        
        return cluster_ids, scores, all_cluster_ids
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max normalization"""
        if not scores or len(scores) == 0:
            return []
        
        scores_arr = np.array(scores)
        min_score = scores_arr.min()
        max_score = scores_arr.max()
        
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        normalized = (scores_arr - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _normalize_scores_zscore(self, scores: List[float]) -> List[float]:
        """Normalize scores using Z-score normalization"""
        if not scores or len(scores) == 0:
            return []
        
        scores_arr = np.array(scores)
        mean = scores_arr.mean()
        std = scores_arr.std()
        
        if std == 0:
            return [0.0] * len(scores)
        
        normalized = (scores_arr - mean) / std
        return normalized.tolist()
    
    def _fusion_combsum(
        self,
        emb_clusters: List[int],
        emb_scores: List[float],
        bm25_clusters: List[int],
        bm25_scores: List[float]
    ) -> List[int]:
        """CombSum fusion: sum normalized scores
        
        Combines embedding-based and BM25 rankings by:
        1. Normalizing both score lists to [0,1] using Min-Max
        2. Summing normalized scores for each cluster
        3. Ranking clusters by combined score (higher is better)
        
        This ensures both methods contribute equally regardless of their
        original score scales.
        """
        # Normalize scores
        emb_scores_norm = self._normalize_scores(emb_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # For embedding distance, lower is better -> invert
        emb_scores_norm = [1.0 - s for s in emb_scores_norm]
        
        # Combine scores
        combined_scores = {}
        
        for cluster_id, score in zip(emb_clusters, emb_scores_norm):
            combined_scores[cluster_id] = combined_scores.get(cluster_id, 0.0) + score
        
        for cluster_id, score in zip(bm25_clusters, bm25_scores_norm):
            combined_scores[cluster_id] = combined_scores.get(cluster_id, 0.0) + score
        
        # Sort by combined score (descending)
        sorted_clusters = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [cluster_id for cluster_id, _ in sorted_clusters]
    
    def _fusion_borda(
        self,
        emb_clusters: List[int],
        emb_scores: List[float],
        bm25_clusters: List[int],
        bm25_scores: List[float]
    ) -> List[int]:
        """Borda Count fusion: rank-based voting"""
        combined_scores = {}
        
        # Embedding ranking (higher rank = better)
        k = len(emb_clusters)
        for rank, cluster_id in enumerate(emb_clusters):
            combined_scores[cluster_id] = combined_scores.get(cluster_id, 0) + (k - rank)
        
        # BM25 ranking
        k = len(bm25_clusters)
        for rank, cluster_id in enumerate(bm25_clusters):
            combined_scores[cluster_id] = combined_scores.get(cluster_id, 0) + (k - rank)
        
        # Sort by combined score (descending)
        sorted_clusters = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [cluster_id for cluster_id, _ in sorted_clusters]
    
    def _fusion_rrf(
        self,
        emb_clusters: List[int],
        emb_scores: List[float],
        bm25_clusters: List[int],
        bm25_scores: List[float],
        k: int = 60
    ) -> List[int]:
        """Reciprocal Rank Fusion (RRF)
        
        Rank-based fusion method that combines rankings without using raw scores.
        For each cluster: RRF_score = 1/(k + rank_embedding) + 1/(k + rank_bm25)
        
        RRF is more robust than score-based methods as it's insensitive to
        score scale differences and outliers.
        """
        combined_scores = {}
        
        # Embedding ranking
        for rank, cluster_id in enumerate(emb_clusters):
            rrf_score = 1.0 / (k + rank + 1)
            combined_scores[cluster_id] = combined_scores.get(cluster_id, 0.0) + rrf_score
        
        # BM25 ranking
        for rank, cluster_id in enumerate(bm25_clusters):
            rrf_score = 1.0 / (k + rank + 1)
            combined_scores[cluster_id] = combined_scores.get(cluster_id, 0.0) + rrf_score
        
        # Sort by combined score (descending)
        sorted_clusters = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [cluster_id for cluster_id, _ in sorted_clusters]
    
    def fuse_rankings(
        self,
        emb_clusters: List[int],
        emb_scores: List[float],
        bm25_clusters: List[int],
        bm25_scores: List[float]
    ) -> List[int]:
        """Fuse two rankings using the configured method"""
        if self.fusion_method == "combsum":
            return self._fusion_combsum(emb_clusters, emb_scores, bm25_clusters, bm25_scores)
        elif self.fusion_method == "borda":
            return self._fusion_borda(emb_clusters, emb_scores, bm25_clusters, bm25_scores)
        elif self.fusion_method == "rrf":
            return self._fusion_rrf(emb_clusters, emb_scores, bm25_clusters, bm25_scores)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def search(self, filename: str, k_clusters: int = 10, verbose: bool = True) -> Dict:
        """
        Search for an image using its ground truth caption
        
        Args:
            filename: Image filename to search for
            k_clusters: Number of top clusters to retrieve
            verbose: Print detailed information
        
        Returns:
            Dictionary with search results and statistics
        """
        # Get caption for this image
        caption = self.filename_to_caption.get(filename)
        if not caption:
            raise ValueError(f"No caption found for {filename}")
        
        # Get image index
        img_idx = self.filename_to_idx.get(filename)
        if img_idx is None:
            raise ValueError(f"Image {filename} not found in index")
        
        # Get real cluster where image is stored
        real_cluster = self.image_to_cluster.get(img_idx)
        if real_cluster is None:
            raise ValueError(f"Image {filename} not found in any cluster")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🔍 SEARCHING FOR: {filename}")
            print(f"📝 Query (caption): {caption[:100]}...")
            print(f"🎯 Real cluster: {real_cluster}")
            print(f"{'='*80}\n")
        
        # Start timing
        start_time = time.perf_counter()
        
        # Encode caption
        query_vector = self._encode_text(caption)
        
        # Get embedding-based ranking
        emb_time_start = time.perf_counter()
        emb_clusters, emb_distances, all_emb_clusters, all_emb_distances = self._get_embedding_ranking(query_vector, k_clusters)
        emb_time = time.perf_counter() - emb_time_start
        
        # Get BM25-based ranking
        bm25_time_start = time.perf_counter()
        bm25_clusters, bm25_scores, all_bm25_clusters = self._get_bm25_ranking(caption, k_clusters)
        bm25_time = time.perf_counter() - bm25_time_start
        
        # Fuse rankings (top-k for display)
        fusion_time_start = time.perf_counter()
        fused_clusters = self.fuse_rankings(emb_clusters, emb_distances, bm25_clusters, bm25_scores)
        
        # Get complete fused ranking for accurate position finding
        # Use the complete rankings we already have from the ranking methods
        all_emb_clusters_limited = all_emb_clusters[:len(all_bm25_clusters)]  # Ensure same length
        all_bm25_clusters_limited = all_bm25_clusters[:len(all_emb_clusters)]
        all_emb_distances_limited = all_emb_distances[:len(all_bm25_clusters)]
        
        # Get BM25 scores for the complete ranking 
        tokenized_query = caption.lower().split()
        bm25_scores_complete = self.bm25.get_scores(tokenized_query)
        all_bm25_scores_sorted = []
        for cid in all_bm25_clusters_limited:
            if cid in self.cluster_id_to_summary_idx:
                score_idx = self.cluster_id_to_summary_idx[cid]
                all_bm25_scores_sorted.append(bm25_scores_complete[score_idx])
            else:
                all_bm25_scores_sorted.append(0.0)
        
        all_fused_clusters = self.fuse_rankings(all_emb_clusters_limited, all_emb_distances_limited, all_bm25_clusters_limited, all_bm25_scores_sorted)
        fusion_time = time.perf_counter() - fusion_time_start
        
        # Find position of real cluster in each ranking (search exhaustively in full lists)
        emb_position = all_emb_clusters.index(real_cluster) + 1 if real_cluster in all_emb_clusters else None
        bm25_position = all_bm25_clusters.index(real_cluster) + 1 if real_cluster in all_bm25_clusters else None
        hybrid_position = all_fused_clusters.index(real_cluster) + 1 if real_cluster in all_fused_clusters else None
        
        if verbose:
            print(f"🔍 Exhaustive position search:")
            print(f"  Embedding: searching in {len(all_emb_clusters)} clusters → Position: {emb_position}")
            print(f"  BM25: searching in {len(all_bm25_clusters)} clusters → Position: {bm25_position}") 
            print(f"  Hybrid: searching in {len(all_fused_clusters)} clusters → Position: {hybrid_position}")
        
        total_time = time.perf_counter() - start_time
        
        # Create result
        result = {
            "filename": filename,
            "caption": caption,
            "real_cluster": real_cluster,
            "embedding_ranking": {
                "clusters": emb_clusters,
                "distances": emb_distances,
                "position": emb_position,
                "time_seconds": emb_time
            },
            "bm25_ranking": {
                "clusters": bm25_clusters,
                "scores": bm25_scores,
                "position": bm25_position,
                "time_seconds": bm25_time
            },
            "hybrid_ranking": {
                "method": self.fusion_method,
                "clusters": fused_clusters[:k_clusters],
                "position": hybrid_position,
                "time_seconds": fusion_time
            },
            "total_time_seconds": total_time
        }
        
        # Print results
        if verbose:
            print(f"⏱️  TIMING:")
            print(f"  Embedding search: {emb_time*1000:.2f} ms")
            print(f"  BM25 search: {bm25_time*1000:.2f} ms")
            print(f"  Fusion ({self.fusion_method}): {fusion_time*1000:.2f} ms")
            print(f"  Total: {total_time*1000:.2f} ms")
            
            print(f"\n📊 EMBEDDING RANKING (top {k_clusters}):")
            for i, (cid, dist) in enumerate(zip(emb_clusters, emb_distances), 1):
                marker = "🎯" if cid == real_cluster else "  "
                print(f"  {marker} #{i}: Cluster {cid} (distance: {dist:.4f})")
            if emb_position:
                if emb_position <= k_clusters:
                    print(f"  ✅ Real cluster position: #{emb_position} (in top-{k_clusters})")
                else:
                    print(f"  ⚠️  Real cluster position: #{emb_position} (NOT in top-{k_clusters})")
            else:
                print(f"  ❌ Real cluster position: NOT FOUND")
            
            print(f"\n📊 BM25 RANKING (top {k_clusters}):")
            for i, (cid, score) in enumerate(zip(bm25_clusters, bm25_scores), 1):
                marker = "🎯" if cid == real_cluster else "  "
                print(f"  {marker} #{i}: Cluster {cid} (score: {score:.4f})")
            if bm25_position:
                if bm25_position <= k_clusters:
                    print(f"  ✅ Real cluster position: #{bm25_position} (in top-{k_clusters})")
                else:
                    print(f"  ⚠️  Real cluster position: #{bm25_position} (NOT in top-{k_clusters})")
            else:
                print(f"  ❌ Real cluster position: NOT FOUND")
            
            print(f"\n🔀 HYBRID RANKING ({self.fusion_method.upper()}, top {k_clusters}):")
            for i, cid in enumerate(fused_clusters[:k_clusters], 1):
                marker = "🎯" if cid == real_cluster else "  "
                print(f"  {marker} #{i}: Cluster {cid}")
            if hybrid_position:
                if hybrid_position <= k_clusters:
                    print(f"  ✅ Real cluster position: #{hybrid_position} (in top-{k_clusters})")
                else:
                    print(f"  ⚠️  Real cluster position: #{hybrid_position} (NOT in top-{k_clusters})")
            else:
                print(f"  ❌ Real cluster position: NOT FOUND")
            
            # Compare with baseline if available
            self._compare_with_baseline(filename, emb_position, hybrid_position, verbose=True)
        
        return result
    
    def _compare_with_baseline(self, filename: str, baseline_position: Optional[int], hybrid_position: Optional[int], verbose: bool = True):
        """Compare with baseline IVF cluster positions"""
        if not self.cluster_positions_path.exists():
            if verbose:
                print(f"\n⚠️  Baseline cluster positions file not found: {self.cluster_positions_path}")
            return
        
        # Load baseline data
        with open(self.cluster_positions_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        
        # Find this image in baseline results
        baseline_result = None
        for result in baseline_data.get("results", []):
            if result.get("filename") == filename:
                baseline_result = result
                break
        
        if not baseline_result:
            if verbose:
                print(f"\n⚠️  No baseline result found for {filename}")
            return
        
        baseline_pos = baseline_result.get("cluster_position")
        baseline_time = baseline_result.get("search_time_seconds", 0)
        
        if verbose:
            print(f"\n📈 COMPARISON WITH BASELINE IVF:")
            print(f"  Baseline (embedding only) position: {baseline_pos}")
            print(f"  Current embedding position: {baseline_position}")
            print(f"  Hybrid ({self.fusion_method}) position: {hybrid_position}")
            
            if hybrid_position and baseline_pos:
                improvement = baseline_pos - hybrid_position
                if improvement > 0:
                    print(f"  ✅ IMPROVEMENT: Hybrid is {improvement} positions better!")
                elif improvement < 0:
                    print(f"  ❌ DEGRADATION: Hybrid is {-improvement} positions worse")
                else:
                    print(f"  ➡️  NO CHANGE: Same position")
            
            print(f"  Baseline search time: {baseline_time*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Hybrid IVF Retrieval with LLM Cluster Summaries")
    parser.add_argument("--dataset", type=str, default="COCO", help="Dataset name")
    parser.add_argument("--llm-name", type=str, default="gemma3_4b", 
                       help="LLM model name (gemma3_4b, mistral_7b)")
    parser.add_argument("--fusion-method", type=str, default="combsum",
                       choices=["combsum", "borda", "rrf"],
                       help="Fusion method for combining rankings")
    parser.add_argument("--filename", type=str, required=True,
                       help="Image filename to search for")
    parser.add_argument("--k-clusters", type=int, default=10,
                       help="Number of top clusters to retrieve")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for inference (cuda/cpu)")
    parser.add_argument("--bm25-variant", type=str, default="okapi",
                       choices=["okapi", "plus", "l", "short"],
                       help="BM25 variant (okapi, plus, l, short). 'short' is optimized for short captions with k1≈1.0, b≈0.4")
    parser.add_argument("--bm25-k1", type=float, default=1.2,
                       help="BM25 k1 parameter (term frequency saturation, recommended: 1.2-2.0)")
    parser.add_argument("--bm25-b", type=float, default=0.75,
                       help="BM25 b parameter (document length normalization, recommended: 0.75)")
    parser.add_argument("--bm25-delta", type=float, default=1.0,
                       help="BM25+ and BM25L delta parameter (lower bound, recommended: 1.0)")
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = HybridIVFRetriever(
        dataset=args.dataset,
        llm_name=args.llm_name,
        fusion_method=args.fusion_method,
        device=args.device,
        bm25_variant=args.bm25_variant,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        bm25_delta=args.bm25_delta
    )
    
    # Search
    result = retriever.search(
        filename=args.filename,
        k_clusters=args.k_clusters,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"✅ Search complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
