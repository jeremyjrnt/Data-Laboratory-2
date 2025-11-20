#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budgeted Cluster Sampling for BLIP Captioning with LLM Cluster Summarization
- Allocate a global caption budget across clusters with diminishing returns
- Select per-cluster samples via FPS seeded at the nearest-to-centroid point
- Generate BLIP captions for selected images
- Use LLM to generate semantic cluster descriptions

Usage:
  python src/indexer/indexer_ivf_llm_cc.py \
      --dataset COCO \
      --budget 10000 \
      --alpha 0.5 \
      --q-min 3 \
      --q-max 200 \
      --metric l2 \
      --outlier-quantile 0.95 \
      --llm-model gemma3:4b \
      --vectordb-dir VectorDBs \
      --images-root data/COCO/images \
      --output VectorDBs/COCO_IVF_KMeans_fps_selection.json

Notes:
- Requires: numpy, torch, pillow, transformers
- metadata.json must contain a top-level key "metadata": list of dicts with
  - "embedding": list/array of floats
  - "filename":  str
"""

import os
import json
import argparse
import logging
import pickle
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import faiss

# ----------------------------- Logging --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("budgeted_cluster_fps")
logger.setLevel(logging.INFO)

# ------------------------- Distance Utilities -------------------------------

def l2_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return ||a - b||_2 for each row of a against vector b."""
    return np.linalg.norm(a - b[None, :], axis=1)

def cosine_dist(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return 1 - cos(a, b) for each row of a against vector b."""
    an = np.linalg.norm(a, axis=1) + eps
    bn = np.linalg.norm(b) + eps
    sim = (a @ b) / (an * bn)
    return 1.0 - sim

def pairwise_l2(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return ||a - v||_2 for each row of a (vectorized)."""
    return np.linalg.norm(a - v[None, :], axis=1)

def pairwise_cosine(a: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return 1 - cos(a, v) for each row of a (vectorized)."""
    an = np.linalg.norm(a, axis=1) + eps
    vn = np.linalg.norm(v) + eps
    sim = (a @ v) / (an * vn)
    return 1.0 - sim

# ---------------------- Budget Allocation (Concave) -------------------------

def allocate_quota_per_cluster(
    cluster_sizes: np.ndarray,
    budget: int = 10_000,
    q_min: int = 3,
    q_max: int = 200,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Allocate integer quotas q_k per cluster with diminishing returns:
      weights ~ n_k ** alpha (0 < alpha < 1), floor = q_min, cap = q_max.

    Ensures sum(q_k) == budget (when possible) via largest-remainder fix.
    Caps q_k by cluster size as well.

    If budget < q_min * K, distribute 1 per cluster until exhausted.
    """
    K = len(cluster_sizes)
    sizes = cluster_sizes.astype(int)

    if K == 0 or budget <= 0:
        return np.zeros(0, dtype=int)

    # If budget too small to give q_min to everyone:
    if budget < q_min * K:
        q = np.zeros(K, dtype=int)
        # distribute one by one across clusters (stable order)
        remaining = budget
        i = 0
        while remaining > 0 and i < K:
            give = min(1, sizes[i])  # cannot exceed cluster size
            q[i] += give
            remaining -= give
            i += 1
        return q

    base = np.minimum(np.full(K, q_min, dtype=int), sizes)  # respect size
    B_prime = budget - base.sum()
    if B_prime < 0:
        # Shouldn't happen due to previous guard, but keep safe
        return base

    # weights = n_k ** alpha (concave)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.power(np.maximum(sizes, 0), alpha).astype(float)
    denom = weights.sum()

    if denom == 0:
        q = base.copy()
    else:
        ideal = base.astype(float) + (B_prime * (weights / denom))
        q = np.rint(ideal).astype(int)

    # Apply caps: per-cluster max and cannot exceed cluster size
    q = np.minimum(q, q_max)
    q = np.minimum(q, sizes)

    # Adjust to match budget exactly (largest remainders)
    diff = budget - q.sum()
    if diff != 0 and denom != 0:
        ideal = base.astype(float) + (B_prime * (weights / denom))
        remainders = ideal - np.floor(ideal)
        order = np.argsort(remainders)[::-1]  # largest remainder first

        # Try to fix diff using remainders ordering
        i = 0
        while diff != 0 and i < K:
            k = order[i]
            if diff > 0:
                add_cap = min(q_max, sizes[k])
                if q[k] < add_cap:
                    q[k] += 1
                    diff -= 1
                else:
                    i += 1
            else:  # diff < 0
                if q[k] > 0:
                    q[k] -= 1
                    diff += 1
                else:
                    i += 1

        # Fallback round-robin pass if needed
        i = 0
        while diff > 0 and i < K:
            cap = min(q_max, sizes[i])
            if q[i] < cap:
                q[i] += 1
                diff -= 1
            i += 1
        i = 0
        while diff < 0 and i < K:
            if q[i] > 0:
                q[i] -= 1
                diff += 1
            i += 1

    return q.astype(int)

# ---------------------------- FPS Selector ----------------------------------

def fps_from_centroid(
    embeds: np.ndarray,
    centroid: np.ndarray,
    q_k: int,
    metric: str = "l2",
) -> List[int]:
    """
    Farthest Point Sampling with central seed:
      1) seed = argmin distance(embedding, centroid)
      2) greedy max-min expansion until q_k points selected

    embeds:  [M, d] array for the cluster
    centroid: [d]
    q_k: number of points to select
    metric: "l2" or "cosine"

    Returns: list of indices (0..M-1) within the cluster block.
    """
    M = embeds.shape[0]
    if q_k >= M:
        return list(range(M))
    if q_k <= 0 or M == 0:
        return []

    # Distances to centroid for seed
    if metric == "cosine":
        d2c = pairwise_cosine(embeds, centroid)
    else:
        d2c = pairwise_l2(embeds, centroid)

    seed = int(np.argmin(d2c))
    selected = [seed]

    # Initialize min distance to the selected set (currently just seed)
    if metric == "cosine":
        min_d = pairwise_cosine(embeds, embeds[seed])
    else:
        min_d = pairwise_l2(embeds, embeds[seed])

    # Greedy max-min selection
    while len(selected) < q_k:
        # Exclude already selected by setting negative distance
        min_d[selected] = -1.0
        nxt = int(np.argmax(min_d))
        selected.append(nxt)

        # Update min distances with the newly added point
        if metric == "cosine":
            dist_new = pairwise_cosine(embeds, embeds[nxt])
        else:
            dist_new = pairwise_l2(embeds, embeds[nxt])

        min_d = np.maximum(min_d, dist_new)

    return selected

# ------------------------- Data Loading Helpers -----------------------------

def load_cluster_artifacts(
    vectordb_dir: Path,
    dataset: str
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray]:
    """
    Load centroids, assignments, metadata, and embeddings from FAISS index.
    Returns:
      centroids:  [K, d]
      assignments: [N,] int cluster id per image
      metadata: list of dicts (length N), each with 'filename' etc.
      embeddings: [N, d] array of all embeddings
    """
    centroids_path = vectordb_dir / f"{dataset}_IVF_KMeans_centroids.npy"
    assignments_path = vectordb_dir / f"{dataset}_IVF_KMeans_assignments.pkl"
    metadata_path   = vectordb_dir / f"{dataset}_VectorDB_metadata.json"
    index_path = vectordb_dir / f"{dataset}_VectorDB.index"

    if not centroids_path.exists():
        raise FileNotFoundError(f"Missing centroids: {centroids_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignments: {assignments_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    centroids = np.load(centroids_path)
    with open(assignments_path, "rb") as f:
        assignments_data = pickle.load(f)
        # Handle both dict format and direct array format
        if isinstance(assignments_data, dict):
            assignments = assignments_data['assignments']
        else:
            assignments = assignments_data

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    metadata = meta_json.get("metadata", [])
    if len(metadata) == 0:
        raise ValueError("metadata.json has empty or missing 'metadata' list")

    if len(assignments) != len(metadata):
        raise ValueError(f"Assignments length {len(assignments)} != metadata length {len(metadata)}")

    # Load FAISS index to extract embeddings
    logger.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    
    # Extract all vectors from the index
    N = index.ntotal
    d = index.d
    embeddings = np.zeros((N, d), dtype=np.float32)
    
    # Reconstruct vectors from FAISS index
    for i in range(N):
        embeddings[i] = index.reconstruct(i)
    
    logger.info(f"Extracted {N} embeddings of dimension {d} from FAISS index")

    return centroids, np.asarray(assignments, dtype=int), metadata, embeddings

# ----------------------------- Main Pipeline --------------------------------

def call_ollama_llm(prompt: str, llm_model: str = None, max_retries: int = 5) -> str:
    """Call Ollama LLM with robust retry mechanism (5 attempts by default)."""
    if llm_model is None:
        llm_model = Config.OLLAMA_MODEL_DEFAULT
    for attempt in range(max_retries):
        try:
            logger.info(f"🤖 LLM Call Attempt {attempt + 1}/{max_retries}")
            
            request_data = {
                "model": llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 8192,  # Context window for long prompts
                }
            }
            
            cmd = [
                "curl", "-s", "-X", "POST",
                "http://localhost:11434/api/generate",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(request_data)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',  # Force UTF-8 encoding pour éviter les erreurs Windows
                errors='replace',  # Remplace les caractères non-décodables au lieu de crasher
                timeout=180  # Augmenté à 3 minutes pour les réponses longues
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Curl failed: {result.stderr}")
            
            response = json.loads(result.stdout)
            llm_response = response.get('response', '').strip()
            
            if not llm_response:
                raise RuntimeError("Empty response from LLM")
            
            logger.info(f"✅ LLM responded successfully (attempt {attempt + 1})")
            return llm_response
            
        except Exception as e:
            logger.warning(f"⚠️ LLM call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 3 + (attempt * 2)  # Délai progressif: 3s, 5s, 7s, 9s, 11s
                logger.info(f"🔄 Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ All {max_retries} LLM call attempts failed")
                return ""
    
    return ""

def generate_blip_caption(image_path: Path, blip_model, blip_processor, device) -> str:
    """Generate BLIP caption for a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = blip_model.generate(
                **inputs,
                max_length=150,
                min_length=30,
                num_beams=8,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False
            )
        
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
        
    except Exception as e:
        logger.error(f"❌ Error generating caption for {image_path}: {e}")
        return "Failed to generate description"

def create_cluster_summary_prompt(cluster_id: int, captions: List[str], cluster_size: int) -> str:
    """Create prompt for LLM to summarize cluster semantic coverage."""
    # Format captions as numbered list
    formatted_captions = "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(captions)])
    
    prompt = f"""You are an expert in analyzing visual data clusters.

Below are text descriptions representing a cluster:

CAPTIONS:
{formatted_captions}

TASK:
Analyze these descriptions to produce a single, natural paragraph that captures the cluster’s full semantic landscape.

Write in fluent, continuous prose (no lists or headings). The text will be used as a document in a retrieval system combining BM25 and visual embeddings, so it must be clear, lexically rich, and semantically dense.

Include:
- Core themes and recurring visual subjects or environments  
- Common attributes, colors, lighting, artistic tone, and composition  
- Typical actions, relations, or interactions among elements  
- Diversity and coherence, including subthemes or outliers  

End with 2–3 sentences summarizing the cluster’s overall meaning, diversity, and visual identity. """
    return prompt

def build_selection(
    dataset: str,
    vectordb_dir: Path,
    images_root: Optional[Path],
    budget: Optional[int] = None,
    alpha: float = 0.5,
    q_min: int = 3,
    q_max: int = 200,
    metric: str = "l2",
    outlier_quantile: Optional[float] = 0.95,
    llm_model: str = None,
    generate_captions: bool = True,
    generate_summaries: bool = True,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    End-to-end:
      - load data
      - allocate quotas
      - per-cluster FPS seeded at centroid-nearest
      - optional outlier filtering by quantile of centroid distance
      - generate BLIP captions for selected images
      - generate LLM summaries for each cluster
      - save json with selections and summaries
    
    If budget is None, defaults to 10% of total number of images.
    """
    logger.info(f"Loading artifacts for dataset '{dataset}' from {vectordb_dir} ...")
    centroids, assignments, metadata, embeds = load_cluster_artifacts(vectordb_dir, dataset)
    K, d = centroids.shape
    N = len(metadata)
    logger.info(f"Loaded {K} clusters, {N} images, embedding dim = {d}")
    
    # Set default budget to 10% of total images if not specified
    if budget is None:
        budget = int(N * 0.1)
        logger.info(f"Budget not specified, using default: {budget} (10% of {N} total images)")
    
    # Load BLIP model if needed
    blip_model = None
    blip_processor = None
    device = None
    if generate_captions:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔄 Loading BLIP model on {device}...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)
        blip_model.eval()
        logger.info("✅ BLIP model loaded")
        
        # Determine images root
        if images_root is None:
            images_root = Path(f"data/{dataset}/images")
        logger.info(f"📂 Images directory: {images_root}")

    # Embeddings were already loaded from FAISS index
    logger.info(f"Using embeddings from FAISS index: {embeds.shape}")

    # Compute cluster sizes
    cluster_sizes = np.bincount(assignments, minlength=K).astype(int)
    logger.info("Allocating quotas with diminishing returns ...")
    quotas = allocate_quota_per_cluster(
        cluster_sizes=cluster_sizes,
        budget=budget,
        q_min=q_min,
        q_max=q_max,
        alpha=alpha,
    )

    logger.info(f"Total allocated = {quotas.sum()} (target budget = {budget})")

    # Prepare metric funcs
    dist_to_centroid_fn = pairwise_l2 if metric == "l2" else pairwise_cosine

    # Prepare output path
    if output_path is None:
        llm_name_sanitized = llm_model.replace(":", "_").replace("/", "_")
        output_path = vectordb_dir / f"{dataset}_IVF_CC_{llm_name_sanitized}.json"
    
    logger.info(f"📁 Output will be saved to: {output_path}")

    # Per-cluster selection
    selections = []
    cluster_summaries = []
    total_selected = 0

    for cid in range(K):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Processing Cluster {cid}/{K}")
        logger.info(f"{'='*80}")
        
        idxs = np.where(assignments == cid)[0]
        n_k = len(idxs)
        q_k = int(quotas[cid])

        if n_k == 0 or q_k == 0:
            selections.append({
                "cluster_id": cid,
                "n_images": n_k,
                "quota": q_k,
                "selected_count": 0,
                "selected_indices_global": [],
                "selected_indices_in_cluster": [],
                "filenames": [],
                "captions": [],
            })
            cluster_summaries.append({
                "cluster_id": cid,
                "n_images": n_k,
                "summary": "Empty cluster",
                "n_captions": 0
            })
            continue

        cluster_embeds = embeds[idxs]           # [n_k, d]
        centroid = centroids[cid]               # [d]
        d2c = dist_to_centroid_fn(cluster_embeds, centroid)

        # Optional anti-outlier filter by quantile of d2c
        mask = np.ones(n_k, dtype=bool)
        if outlier_quantile is not None:
            if 0.5 <= outlier_quantile < 1.0:
                thr = np.quantile(d2c, outlier_quantile)
                mask = d2c <= thr
            else:
                logger.warning(f"Invalid outlier_quantile={outlier_quantile}; ignored.")
        filtered_embeds = cluster_embeds[mask]
        filtered_idxs = idxs[mask]
        n_f = len(filtered_embeds)

        # If filtering removed too much, fall back
        if n_f == 0:
            filtered_embeds = cluster_embeds
            filtered_idxs = idxs
            n_f = n_k

        # Adjust quota if needed
        q_eff = min(q_k, n_f)

        # Run FPS from centroid
        local_sel = fps_from_centroid(
            embeds=filtered_embeds,
            centroid=centroid,
            q_k=q_eff,
            metric=metric,
        )  # indices local to filtered_embeds

        global_sel = filtered_idxs[local_sel]
        filenames = [metadata[i]["filename"] for i in global_sel]
        
        # Generate BLIP captions if requested
        captions = []
        if generate_captions and blip_model is not None:
            logger.info(f"🖼️  Generating BLIP captions for {len(filenames)} selected images...")
            for filename in filenames:
                image_path = images_root / filename
                if image_path.exists():
                    caption = generate_blip_caption(image_path, blip_model, blip_processor, device)
                    captions.append(caption)
                    logger.info(f"   ✅ {filename}: {caption[:70]}...")
                else:
                    logger.warning(f"   ⚠️ Image not found: {image_path}")
                    captions.append("Image file not found")
        
        # Generate LLM cluster summary if requested
        cluster_summary = ""
        if generate_summaries and len(captions) > 0:
            logger.info(f"🤖 Generating LLM summary for cluster {cid}...")
            summary_prompt = create_cluster_summary_prompt(cid, captions, n_k)
            cluster_summary = call_ollama_llm(summary_prompt, llm_model=llm_model)
            if cluster_summary:
                logger.info(f"✅ Cluster {cid} summary: {cluster_summary[:100]}...")
            else:
                logger.warning(f"⚠️ Failed to generate summary for cluster {cid}")
                cluster_summary = f"Cluster with {n_k} images, {len(captions)} captions generated"

        selections.append({
            "cluster_id": cid,
            "n_images": n_k,
            "quota": q_k,
            "selected_count": len(global_sel),
            "selected_indices_global": global_sel.tolist(),
            "selected_indices_in_cluster": local_sel,  # local to filtered set
            "filenames": filenames,
            "captions": captions,
        })
        
        cluster_summaries.append({
            "cluster_id": cid,
            "n_images": n_k,
            "summary": cluster_summary,
            "n_captions": len(captions),
            "selected_filenames": filenames
        })
        
        total_selected += len(global_sel)
        
        # Save incrementally after each cluster
        result = {
            "dataset": dataset,
            "budget": budget,
            "alpha": alpha,
            "q_min": q_min,
            "q_max": q_max,
            "metric": metric,
            "outlier_quantile": outlier_quantile,
            "llm_model": llm_model,
            "n_clusters": K,
            "n_images": N,
            "allocated_total": int(quotas.sum()),
            "selected_total": int(total_selected),
            "quotas": quotas.tolist(),
            "selections": selections,
            "cluster_summaries": cluster_summaries,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved progress: {cid + 1}/{K} clusters processed")

    # Final save
    logger.info(f"✅ Complete: selected_total={total_selected}, allocated_total={quotas.sum()}, budget={budget}")
    logger.info(f"📁 Final results saved to: {output_path}")

    return result

# --------------------------------- CLI --------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Allocate BLIP caption budget across clusters, generate captions, and create LLM cluster summaries."
    )
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., COCO, Flickr, VizWiz)")
    p.add_argument("--vectordb-dir", type=str, default=None, help="Directory with centroids/assignments/metadata (default: from Config)")
    p.add_argument("--images-root", type=str, default=None, help="Root folder of images (default: data/{dataset}/images)")
    p.add_argument("--budget", type=int, default=None, help="Global number of captions to allocate (default: 10%% of total images)")
    p.add_argument("--alpha", type=float, default=0.5, help="Concavity exponent in (0,1); e.g. 0.5 = sqrt")
    p.add_argument("--q-min", type=int, default=3, help="Per-cluster floor")
    p.add_argument("--q-max", type=int, default=200, help="Per-cluster cap")
    p.add_argument("--metric", type=str, default="l2", choices=["l2", "cosine"], help="Distance metric")
    p.add_argument("--outlier-quantile", type=float, default=0.95,
                   help="Keep points with dist-to-centroid <= quantile; set >=1 to disable")
    p.add_argument("--llm-model", type=str, default="gemma3:4b", help="LLM model for cluster summaries")
    p.add_argument("--no-captions", action="store_true", help="Skip BLIP caption generation")
    p.add_argument("--no-summaries", action="store_true", help="Skip LLM cluster summary generation")
    p.add_argument("--output", type=str, default=None, help="JSON output path")
    return p.parse_args()

def main():
    args = parse_args()
    vectordb_dir = Path(args.vectordb_dir)
    images_root = Path(args.images_root) if args.images_root else None
    output_path = Path(args.output) if args.output else None

    if args.outlier_quantile >= 1.0:
        outlier_q = None
    else:
        outlier_q = args.outlier_quantile

    build_selection(
        dataset=args.dataset,
        vectordb_dir=vectordb_dir,
        images_root=images_root,
        budget=args.budget,
        alpha=args.alpha,
        q_min=args.q_min,
        q_max=args.q_max,
        metric=args.metric,
        outlier_quantile=outlier_q,
        llm_model=args.llm_model,
        generate_captions=not args.no_captions,
        generate_summaries=not args.no_summaries,
        output_path=output_path,
    )

if __name__ == "__main__":
    main()