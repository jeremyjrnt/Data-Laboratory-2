#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IVF (Inverted File) Indexer with FAISS KMeans + Verbose Logging
- Clustering with faiss.Kmeans (CPU/GPU)
- IVF training on training vectors (not centroids)
- Cosine (IP) or L2 metrics
- Detailed progress and memory logs
"""

import os
import json
import faiss
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import psutil
import time
import math
import subprocess
import re
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


# -------------------------- Logging & Utils --------------------------

def log_info(message: str):
    """Utility for timestamped console logs."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def memory_usage_gb():
    """Returns current memory usage in GB."""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return float("nan")


def human_int(n: int) -> str:
    return f"{n:,}"


# -------------------------- Core Indexer --------------------------

class IVFIndexerAL:
    """
    IVF Indexer using FAISS K-means for efficient vector search with detailed logging.
    Includes Active Learning with BLIP descriptions and LLM-based reassignment.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        n_init: int = 10,
        metric: str = "ip",           # 'ip' (cosine) or 'l2'
        use_gpu: bool = False,
        kmeans_iters: int = 25,
        target_nprobe: Optional[int] = None,
        reconstruct_batch: int = 8192,
        # Active Learning parameters
        al_iterations: int = 3,
        al_n_boundary_points: int = 1000,
        al_llm_model: str = "gemma3:4b",
        dataset_name: str = "COCO",
        images_dir: Optional[str] = None
    ):
        self.n_clusters = n_clusters
        self.n_init = int(n_init)
        self.metric = metric.lower()
        assert self.metric in {"ip", "l2"}, "metric must be 'ip' or 'l2'"
        self.use_gpu = bool(use_gpu)
        self.kmeans_iters = int(kmeans_iters)
        self.target_nprobe = target_nprobe
        self.reconstruct_batch = int(reconstruct_batch)
        
        # Active Learning parameters
        self.al_iterations = al_iterations
        self.al_n_boundary_points = al_n_boundary_points
        self.al_llm_model = al_llm_model
        self.dataset_name = dataset_name
        self.images_dir = images_dir

        self.index: Optional[faiss.Index] = None
        self.centroids: Optional[np.ndarray] = None
        self.assignments: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.dimension: Optional[int] = None
        
        # BLIP model for image captioning
        self.blip_processor = None
        self.blip_model = None
        self.cluster_descriptions: Dict[int, str] = {}

        self._gpu_res = None  # lazily created

    # --------------- GPU helpers ---------------

    def _ensure_gpu_res(self):
        if self._gpu_res is None:
            self._gpu_res = faiss.StandardGpuResources()
        return self._gpu_res

    def _maybe_to_gpu(self, index: faiss.Index) -> faiss.Index:
        if not self.use_gpu:
            return index
        res = self._ensure_gpu_res()
        return faiss.index_cpu_to_gpu(res, 0, index)

    def _maybe_to_cpu_for_write(self, index: faiss.Index) -> faiss.Index:
        # Try to move GPU index to CPU for saving; no-op if already CPU
        try:
            return faiss.index_gpu_to_cpu(index)
        except Exception:
            return index

    # --------------- Metric & Quantizer ---------------

    def _make_flat(self, d: int) -> faiss.Index:
        if self.metric == "ip":
            return faiss.IndexFlatIP(d)
        else:
            return faiss.IndexFlatL2(d)

    def _metric_type(self) -> int:
        return faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2

    # --------------- BLIP & LLM for Active Learning ---------------

    def _call_ollama_api(self, prompt: str, model: str = None) -> str:
        """
        Call Ollama API locally to get LLM response.
        
        Args:
            prompt: The prompt to send to the LLM
            model: The model to use (defaults to self.al_llm_model)
        
        Returns:
            LLM response as string, empty string on error
        """
        if model is None:
            model = self.al_llm_model
            
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.ConnectionError:
            log_info(f"⚠️ Cannot connect to Ollama API at {url}. Please ensure Ollama is running.")
            return ""
        except requests.exceptions.Timeout:
            log_info(f"⚠️ Ollama API timeout for prompt")
            return ""
        except requests.exceptions.RequestException as e:
            log_info(f"⚠️ Ollama API error: {e}")
            return ""
        except Exception as e:
            log_info(f"⚠️ Unexpected error calling Ollama API: {e}")
            return ""

    def _load_blip_model(self):
        """Load BLIP model for image captioning."""
        if self.blip_processor is None:
            log_info("Loading BLIP model for image captioning...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
            self.blip_model = self.blip_model.to(device)
            log_info(f"✅ BLIP model loaded on {device}")

    def _describe_image(self, image_path: str) -> str:
        """Generate a description for an image using BLIP."""
        try:
            image = Image.open(image_path).convert('RGB')
            device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
            
            inputs = self.blip_processor(image, return_tensors="pt").to(device)
            out = self.blip_model.generate(**inputs, max_length=50)
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return description
        except Exception as e:
            log_info(f"⚠️ Error describing image {image_path}: {e}")
            return "No description available"

    def _describe_cluster(self, cluster_id: int, vectors: np.ndarray, sample_size: int = 10) -> str:
        """
        Describe a cluster by sampling and describing images from it.
        
        Args:
            cluster_id: ID of the cluster
            vectors: All vectors
            sample_size: Number of images to sample per cluster
        """
        # Find all points in this cluster
        cluster_indices = np.where(self.assignments == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            return "Empty cluster"
        
        # Sample random images from cluster
        sample_indices = np.random.choice(
            cluster_indices, 
            size=min(sample_size, len(cluster_indices)), 
            replace=False
        )
        
        descriptions = []
        for idx in sample_indices:
            if idx < len(self.metadata):
                image_filename = self.metadata[idx].get('image_filename', '')
                if image_filename and self.images_dir:
                    image_path = os.path.join(self.images_dir, image_filename)
                    if os.path.exists(image_path):
                        desc = self._describe_image(image_path)
                        descriptions.append(desc)
        
        if descriptions:
            # Combine descriptions
            combined = " | ".join(descriptions[:5])  # Limit to 5 descriptions
            return f"Cluster {cluster_id}: {combined}"
        else:
            return f"Cluster {cluster_id}: No images available"

    def _describe_all_clusters(self, vectors: np.ndarray):
        """Describe all clusters using BLIP."""
        log_info(f"📝 Describing all {self.n_clusters} clusters with BLIP...")
        self._load_blip_model()
        
        for cluster_id in range(self.n_clusters):
            desc = self._describe_cluster(cluster_id, vectors, sample_size=10)
            self.cluster_descriptions[cluster_id] = desc
            
            if (cluster_id + 1) % 10 == 0:
                log_info(f"  Described {cluster_id + 1}/{self.n_clusters} clusters")
        
        log_info("✅ All clusters described")

    def _find_boundary_points(self, vectors: np.ndarray, n_points: int = 1000) -> np.ndarray:
        """
        Find boundary points: points farthest from their assigned centroid.
        
        Args:
            vectors: All vectors
            n_points: Number of boundary points to find
        
        Returns:
            Indices of boundary points
        """
        log_info(f"🔍 Finding {n_points} boundary points (farthest from centroids)...")
        
        # Calculate distance from each point to its assigned centroid
        distances = np.zeros(len(vectors))
        for i in range(len(vectors)):
            assigned_cluster = self.assignments[i]
            centroid = self.centroids[assigned_cluster]
            
            if self.metric == "ip":
                # For cosine similarity (IP), lower is farther
                distances[i] = -np.dot(vectors[i], centroid)
            else:
                # For L2, higher is farther
                distances[i] = np.linalg.norm(vectors[i] - centroid)
        
        # Get indices of points with largest distances
        boundary_indices = np.argsort(distances)[-n_points:]
        
        log_info(f"✅ Found {len(boundary_indices)} boundary points")
        return boundary_indices

    def _llm_reassign_point(self, point_idx: int, vectors: np.ndarray) -> int:
        """
        Use LLM to decide which cluster a boundary point should belong to.
        
        Args:
            point_idx: Index of the point to reassign
            vectors: All vectors
        
        Returns:
            New cluster assignment
        """
        current_cluster = self.assignments[point_idx]
        
        # Get image description
        image_filename = self.metadata[point_idx].get('image_filename', '') if point_idx < len(self.metadata) else ''
        point_description = "No description"
        
        if image_filename and self.images_dir:
            image_path = os.path.join(self.images_dir, image_filename)
            if os.path.exists(image_path):
                point_description = self._describe_image(image_path)
        
        # Find nearest clusters (top 3)
        point_vec = vectors[point_idx]
        distances = []
        for cluster_id in range(self.n_clusters):
            centroid = self.centroids[cluster_id]
            if self.metric == "ip":
                dist = -np.dot(point_vec, centroid)
            else:
                dist = np.linalg.norm(point_vec - centroid)
            distances.append((cluster_id, dist))
        
        # Sort by distance and get top 3 candidates
        if self.metric == "ip":
            distances.sort(key=lambda x: x[1])  # Lower is better for IP
        else:
            distances.sort(key=lambda x: x[1])  # Lower is better for L2
        
        candidate_clusters = [d[0] for d in distances[:3]]
        
        # Prepare LLM prompt
        prompt = f"""Given an image described as: "{point_description}"

This image is currently assigned to Cluster {current_cluster}.
Here are the candidate clusters and their descriptions:

"""
        for cluster_id in candidate_clusters:
            cluster_desc = self.cluster_descriptions.get(cluster_id, "No description")
            prompt += f"- {cluster_desc}\n"
        
        prompt += f"""
Based on the image description and cluster descriptions, which cluster (by number) should this image belong to?
Respond with ONLY the cluster number (e.g., just "5" or "12"), nothing else."""

        # Call LLM via Ollama API
        answer = self._call_ollama_api(prompt)
        
        if answer:
            # Extract cluster number from response
            numbers = re.findall(r'\d+', answer)
            if numbers:
                new_cluster = int(numbers[0])
                if 0 <= new_cluster < self.n_clusters:
                    return new_cluster
        
        # If API call failed or parsing failed, return current cluster
        return current_cluster

    def _active_learning_iteration(self, vectors: np.ndarray, iteration: int):
        """
        Perform one iteration of active learning: find boundary points and reassign them.
        
        Args:
            vectors: All vectors
            iteration: Current iteration number
        """
        log_info(f"🔄 Active Learning Iteration {iteration + 1}/{self.al_iterations}")
        
        # Find boundary points
        boundary_indices = self._find_boundary_points(vectors, self.al_n_boundary_points)
        
        # Reassign each boundary point using LLM
        reassignments = 0
        for i, point_idx in enumerate(boundary_indices):
            old_cluster = self.assignments[point_idx]
            new_cluster = self._llm_reassign_point(point_idx, vectors)
            
            if new_cluster != old_cluster:
                self.assignments[point_idx] = new_cluster
                reassignments += 1
            
            if (i + 1) % 100 == 0:
                log_info(f"  Processed {i + 1}/{len(boundary_indices)} boundary points | Reassignments: {reassignments}")
        
        log_info(f"✅ Iteration {iteration + 1} complete | Total reassignments: {reassignments}")
        
        # Rebuild centroids based on new assignments
        self._recompute_centroids(vectors)

    def _recompute_centroids(self, vectors: np.ndarray):
        """Recompute centroids based on current assignments."""
        log_info("Recomputing centroids based on new assignments...")
        new_centroids = np.zeros((self.n_clusters, self.dimension), dtype=np.float32)
        
        for cluster_id in range(self.n_clusters):
            cluster_points = vectors[self.assignments == cluster_id]
            if len(cluster_points) > 0:
                new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
            else:
                # Keep old centroid if cluster is empty
                new_centroids[cluster_id] = self.centroids[cluster_id]
        
        # Normalize if using IP metric
        if self.metric == "ip":
            faiss.normalize_L2(new_centroids)
        
        self.centroids = new_centroids
        log_info("✅ Centroids recomputed")

    def _perform_active_learning(self, vectors: np.ndarray):
        """
        Perform active learning with BLIP descriptions and LLM reassignment.
        
        Args:
            vectors: All vectors
        """
        if not self.images_dir or self.al_iterations <= 0:
            log_info("⏭️ Skipping Active Learning (no images_dir or al_iterations=0)")
            return
        
        log_info("="*60)
        log_info("🤖 Starting Active Learning with BLIP + LLM")
        log_info(f"Iterations: {self.al_iterations} | Boundary points per iteration: {self.al_n_boundary_points}")
        log_info(f"LLM model: {self.al_llm_model}")
        log_info("="*60)
        
        # Step 1: Describe all clusters
        self._describe_all_clusters(vectors)
        
        # Step 2: Iterate active learning
        for iteration in range(self.al_iterations):
            self._active_learning_iteration(vectors, iteration)
        
        log_info("="*60)
        log_info("✅ Active Learning completed")
        log_info("="*60)

    # --------------- KMeans (FAISS) ---------------

    def _kmeans_faiss(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train FAISS KMeans on x and return (centroids, assignments).
        Assignments computed via 1-NN to centroids with same metric.
        """
        d = x.shape[1]
        spherical = (self.metric == "ip")  # cosine-like when vectors are L2-normalized

        km = faiss.Kmeans(
            d,
            k,
            niter=self.kmeans_iters,
            nredo=self.n_init,
            verbose=False,
            spherical=spherical
        )

        if self.use_gpu:
            # GPU KMeans training
            km.train(x, self._ensure_gpu_res())
        else:
            km.train(x)

        centroids = km.centroids.astype("float32")

        # Compute assignments by searching nearest centroid
        qtz = self._make_flat(d)
        qtz = self._maybe_to_gpu(qtz)
        qtz.add(centroids)
        _, I = qtz.search(x, 1)  # shape (N,1)
        assignments = I.reshape(-1).astype(np.int32)
        return centroids, assignments

    # --------------- Build IVF ---------------

    def build_index(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Build the IVF index using FAISS KMeans with step-by-step progress info.
        """
        start_time = time.time()
        N, D = vectors.shape
        self.dimension = D

        if self.n_clusters is None:
            self.n_clusters = max(1, int(round(math.sqrt(N))))
        self.n_clusters = max(1, min(self.n_clusters, N))

        log_info("🚀 Starting IVF index construction (FAISS KMeans)")
        log_info(f"N={human_int(N)}  D={D}  K={self.n_clusters}  nredo={self.n_init}  "
                 f"iters={self.kmeans_iters}  metric={self.metric.upper()}  gpu={self.use_gpu}")
        log_info(f"Initial RSS: {memory_usage_gb():.2f} GB")

        # Prepare vectors
        x = vectors.astype("float32", copy=False)
        if self.metric == "ip":
            log_info("Normalizing vectors (L2) for cosine/IP...")
            faiss.normalize_L2(x)
            log_info("✅ Normalization done")

        # KMeans
        log_info("Clustering with faiss.Kmeans...")
        self.centroids, self.assignments = self._kmeans_faiss(x, self.n_clusters)
        log_info("✅ KMeans complete")
        log_info(f"Memory RSS after KMeans: {memory_usage_gb():.2f} GB")

        # IVF index (Flat quantizer with chosen metric)
        log_info("Building IVF index...")
        quant_cpu = self._make_flat(D)
        metric_type = self._metric_type()

        # CPU IVF (train/add works on CPU; we can move to GPU after)
        ivf_cpu = faiss.IndexIVFFlat(quant_cpu, D, self.n_clusters, metric_type)

        log_info("Training IVF on training vectors...")
        ivf_cpu.train(x)
        log_info("✅ IVF trained")

        log_info("Adding vectors to IVF...")
        ivf_cpu.add(x)
        log_info(f"✅ IVF add complete | ntotal={human_int(ivf_cpu.ntotal)}")
        log_info(f"Memory RSS after IVF: {memory_usage_gb():.2f} GB")

        # Optionally move the trained+filled IVF to GPU for querying
        self.index = self._maybe_to_gpu(ivf_cpu)

        # nprobe heuristic if not provided
        if self.target_nprobe is None:
            nprobe = max(1, min(64, self.n_clusters // 50))
        else:
            nprobe = int(self.target_nprobe)
        try:
            self.index.nprobe = nprobe  # works for IVF indexes
        except Exception:
            pass
        log_info(f"nprobe set to {nprobe}")

        # Metadata
        self.metadata = metadata if metadata is not None else [{"id": i} for i in range(N)]

        elapsed = time.time() - start_time
        log_info(f"⏱️ Total build time: {elapsed/60:.2f} minutes")
        log_info(f"Final RSS: {memory_usage_gb():.2f} GB")
        
        # Active Learning phase
        self._perform_active_learning(x)

    # --------------- Compute cluster stats ---------------

    def _compute_cluster_statistics(self) -> Dict:
        """Compute and log detailed cluster statistics."""
        if self.assignments is None:
            return {}

        unique, counts = np.unique(self.assignments, return_counts=True)
        stats = {
            "n_clusters": int(self.n_clusters),
            "n_vectors": int(len(self.assignments)),
            "cluster_sizes": {
                "min": int(np.min(counts)),
                "max": int(np.max(counts)),
                "mean": float(np.mean(counts)),
                "median": float(np.median(counts)),
                "std": float(np.std(counts)),
            }
        }
        log_info(
            "📊 Cluster size distribution — "
            f"min: {stats['cluster_sizes']['min']}, "
            f"max: {stats['cluster_sizes']['max']}, "
            f"mean: {stats['cluster_sizes']['mean']:.1f}, "
            f"std: {stats['cluster_sizes']['std']:.1f}"
        )
        return stats

    # --------------- Save / Load ---------------

    def save_index(self, output_dir: str, index_name: str):
        """
        Save the IVF index, centroids, assignments, and statistics with progress info.
        """
        os.makedirs(output_dir, exist_ok=True)
        log_info("💾 Saving FAISS IVF index and metadata...")

        cpu_index = self._maybe_to_cpu_for_write(self.index)

        index_path = os.path.join(output_dir, f"{index_name}_IVF.index")
        faiss.write_index(cpu_index, index_path)
        log_info(f"Index saved: {index_path}")

        centroids_path = os.path.join(output_dir, f"{index_name}_centroids.npy")
        np.save(centroids_path, self.centroids)
        log_info(f"Centroids saved: {centroids_path}")

        assignments_path = os.path.join(output_dir, f"{index_name}_assignments.npy")
        np.save(assignments_path, self.assignments)
        log_info(f"Assignments saved: {assignments_path}")

        metadata_dict = {
            "index_name": f"{index_name}_IVF",
            "index_type": "IVF",
            "dimension": self.dimension,
            "n_vectors": int(cpu_index.ntotal),
            "n_clusters": int(self.n_clusters),
            "n_init": int(self.n_init),
            "kmeans_iters": int(self.kmeans_iters),
            "metric": self.metric.upper(),
            "nprobe": int(getattr(self.index, "nprobe", None) or 0),
            "al_iterations": self.al_iterations,
            "al_n_boundary_points": self.al_n_boundary_points,
            "al_llm_model": self.al_llm_model,
            "metadata": self.metadata,
        }
        metadata_path = os.path.join(output_dir, f"{index_name}_IVF_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        log_info(f"Metadata saved: {metadata_path}")
        
        # Save cluster descriptions
        if self.cluster_descriptions:
            cluster_desc_path = os.path.join(output_dir, f"{index_name}_cluster_descriptions.json")
            with open(cluster_desc_path, "w", encoding="utf-8") as f:
                json.dump(self.cluster_descriptions, f, indent=2, ensure_ascii=False)
            log_info(f"Cluster descriptions saved: {cluster_desc_path}")

        stats = self._compute_cluster_statistics()
        stats_path = os.path.join(output_dir, f"{index_name}_cluster_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        log_info(f"Cluster statistics saved: {stats_path}")
        log_info("✅ All files successfully saved!")

    @staticmethod
    def _reconstruct_all_from_index(
        index: faiss.Index,
        batch_size: int = 8192
    ) -> np.ndarray:
        """
        Reconstruct all vectors from a FAISS index (generic fallback).
        Uses per-id reconstruct in batches.
        """
        n = index.ntotal
        d = index.d
        log_info(f"Reconstructing {human_int(n)} vectors (d={d}) from FAISS index in batches of {batch_size}…")
        x = np.empty((n, d), dtype="float32")
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for i in range(start, end):
                x[i] = index.reconstruct(i)
            if (start // batch_size) % 10 == 0:
                log_info(f"  … {human_int(end)}/{human_int(n)} reconstructed")
        return x

    def load_vector_db(
        self,
        db_index_path: str,
        metadata_path: str,
        vectors_npy: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load embeddings + metadata. Prefer a .npy if provided; otherwise reconstruct from FAISS index.
        """
        if vectors_npy and os.path.exists(vectors_npy):
            log_info(f"Loading embeddings from .npy: {vectors_npy}")
            x = np.load(vectors_npy).astype("float32", copy=False)
            log_info(f"✅ Loaded {human_int(x.shape[0])} vectors (d={x.shape[1]}) from .npy")
        else:
            log_info(f"Loading FAISS index: {db_index_path}")
            if not os.path.exists(db_index_path):
                raise FileNotFoundError(f"VectorDB not found at {db_index_path}")
            existing_index = faiss.read_index(db_index_path)

            # Reconstruct vectors (generic path)
            x = self._reconstruct_all_from_index(existing_index, batch_size=self.reconstruct_batch)
            log_info(f"✅ Reconstructed {human_int(x.shape[0])} vectors (d={x.shape[1]}) from FAISS index")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            md_raw = json.load(f)
        metadata = md_raw.get("metadata", [])
        log_info(f"Metadata entries loaded: {human_int(len(metadata))}")

        if len(metadata) and len(metadata) != x.shape[0]:
            log_info(f"⚠️ Metadata length {len(metadata)} != vectors count {x.shape[0]} — proceeding anyway")

        return x, metadata


# -------------------------- CLI --------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an IVF index with FAISS K-means from a VectorDB (with logging)"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["COCO", "Flickr", "VizWiz"])
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=10, help="FAISS KMeans nredo")
    parser.add_argument("--kmeans_iters", type=int, default=25, help="FAISS KMeans niter")
    parser.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"], help="Similarity metric")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for KMeans and/or IVF")
    parser.add_argument("--nprobe", type=int, default=None, help="IVF nprobe (default heuristic if omitted)")
    parser.add_argument("--output_dir", type=str, default="VectorDBs")

    # Inputs: FAISS index + metadata (default layout), optional .npy for raw vectors
    parser.add_argument("--db_index_path", type=str, default=None,
                        help="Path to existing FAISS flat index to reconstruct vectors from")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Path to JSON metadata (expects key 'metadata')")
    parser.add_argument("--vectors_npy", type=str, default=None,
                        help="Optional .npy with raw embeddings (preferred over reconstruct)")

    # Reconstruct batching
    parser.add_argument("--reconstruct_batch", type=int, default=8192)
    
    # Active Learning parameters
    parser.add_argument("--al_iterations", type=int, default=3, 
                        help="Number of active learning iterations")
    parser.add_argument("--al_n_boundary_points", type=int, default=1000,
                        help="Number of boundary points to reassign per iteration")
    parser.add_argument("--al_llm_model", type=str, default="gemma3:4b",
                        help="LLM model to use for reassignment decisions")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directory containing images for BLIP descriptions")

    return parser.parse_args()


def main():
    args = parse_args()

    # Default paths (compatible with your previous layout)
    dataset = args.dataset
    db_index_path = args.db_index_path or f"VectorDBs/{dataset}_VectorDB.index"
    metadata_path = args.metadata_path or f"VectorDBs/{dataset}_VectorDB_metadata.json"
    
    # Default images directory if not specified
    images_dir = args.images_dir
    if images_dir is None:
        images_dir = f"data/{dataset}/images"

    if not os.path.exists(metadata_path):
        log_info(f"❌ Error: Metadata not found at {metadata_path}")
        return
    if args.vectors_npy is None and not os.path.exists(db_index_path):
        log_info(f"❌ Error: VectorDB not found at {db_index_path} (or provide --vectors_npy)")
        return

    indexer = IVFIndexerAL(
        n_clusters=args.n_clusters,
        n_init=args.n_init,
        metric=args.metric,
        use_gpu=args.use_gpu,
        kmeans_iters=args.kmeans_iters,
        target_nprobe=args.nprobe,
        reconstruct_batch=args.reconstruct_batch,
        al_iterations=args.al_iterations,
        al_n_boundary_points=args.al_n_boundary_points,
        al_llm_model=args.al_llm_model,
        dataset_name=dataset,
        images_dir=images_dir
    )

    # Load data
    vectors, metadata = indexer.load_vector_db(
        db_index_path=db_index_path,
        metadata_path=metadata_path,
        vectors_npy=args.vectors_npy
    )

    # Build IVF
    indexer.build_index(vectors, metadata)

    # Save
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    indexer.save_index(output_dir, dataset)

    log_info("=" * 60)
    log_info("🎉 IVF index construction completed successfully!")
    log_info("=" * 60)


if __name__ == "__main__":
    main()