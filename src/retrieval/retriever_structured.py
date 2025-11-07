#!/usr/bin/env python3
"""
Simple Image Retrieval using CLIP and FAISS.
Retrieves top-k images from a text prompt using a pre-built vector database.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration

# ------------------------------- Logging -----------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("image_retrieval")
logger.setLevel(logging.DEBUG)

# Suppress specific loggers that are too verbose
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# ------------------------------- Helpers -----------------------------------
def _detect_faiss_metric(index) -> str:
    """Detect whether the FAISS index uses Inner Product or L2 distance."""
    if hasattr(index, "metric_type"):
        return "ip" if index.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"
    name = type(index).__name__.lower()
    if "ip" in name:
        return "ip"
    if "l2" in name:
        return "l2"
    return "ip"

def _standardize_scores(faiss_scores: np.ndarray, metric: str) -> np.ndarray:
    """Convert FAISS raw values into 'higher is better' scores."""
    return -faiss_scores if metric == "l2" else faiss_scores

# ------------------------ Core Retriever Class -----------------------------
class ImageRetriever:
    """
    Image retrieval system using CLIP embeddings and FAISS index.
    
    Args:
        vectordb_name: Name of the vector database (e.g., 'COCO_VectorDB')
        vectordb_dir: Directory containing the .index and _metadata.json files
    """
    
    def __init__(
        self,
        vectordb_name: str,
        vectordb_dir: str = "VectorDBs",
        llm_model: str = "gemma3:27b",
        llm_temp: float = 0.1,
        llm_top_p: float = 0.9,
        llm_timeout: int = 200
    ):
        self.vectordb_name = vectordb_name
        self.vectordb_dir = Path(vectordb_dir)
        self.dataset_name = vectordb_name.replace("_VectorDB", "")
        self.llm_model = llm_model
        self.llm_temp = llm_temp
        self.llm_top_p = llm_top_p
        self.llm_timeout = llm_timeout
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        logger.info("🔄 Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        logger.info(f"✅ CLIP ready on {self.device}")
        
        # Load BLIP model for image captioning
        logger.info("🔄 Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        self.blip_model.eval()
        logger.info(f"✅ BLIP ready on {self.device}")
        
        # Load FAISS index and metadata
        self._load_vectordb()
        
        # Load dataset metadata (for true captions and baseline_rank)
        self._load_dataset_metadata()
        
        # Detect metric and IDMap usage
        self.index_metric = _detect_faiss_metric(self.faiss_index)
        logger.info(f"📐 FAISS metric: {self.index_metric.upper()}")
        self.uses_idmap = isinstance(self.faiss_index, (faiss.IndexIDMap, faiss.IndexIDMap2))
        logger.info(f"🆔 Index uses IDMap: {self.uses_idmap}")
        
        # Build id to metadata mapping
        self._id_to_meta = {}
        for i, m in enumerate(self.metadata):
            eid = int(m.get("embedding_id", i))
            self._id_to_meta[eid] = m
        
        # Set images directory
        self.images_dir = Path("data") / self.dataset_name / "images"
        logger.info(f"📂 Image dir: {self.images_dir} (exists: {self.images_dir.exists()})")
    
    def _load_vectordb(self):
        """Load FAISS index and metadata from disk."""
        index_path = self.vectordb_dir / f"{self.vectordb_name}.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(f"📊 FAISS index loaded: {self.faiss_index.ntotal} vectors")
        
        metadata_path = self.vectordb_dir / f"{self.vectordb_name}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_json = json.load(f)
        
        if isinstance(metadata_json, list):
            self.metadata = metadata_json
        elif isinstance(metadata_json, dict) and "metadata" in metadata_json:
            self.metadata = metadata_json["metadata"]
        else:
            raise ValueError("Invalid metadata format for *_metadata.json")
        
        if self.faiss_index.ntotal != len(self.metadata):
            logger.warning("⚠️ Mismatch: FAISS has %d vectors, metadata has %d entries",
                          self.faiss_index.ntotal, len(self.metadata))
    
    def _load_dataset_metadata(self):
        """Load dataset metadata to get true captions and baseline ranks."""
        dataset_metadata_path = Path("data") / self.dataset_name / f"{self.dataset_name}_metadata.json"
        self.dataset_metadata = {}
        
        if dataset_metadata_path.exists():
            try:
                with open(dataset_metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Build filename to metadata mapping
                if isinstance(data, list):
                    for item in data:
                        filename = item.get("filename")
                        if filename:
                            self.dataset_metadata[filename] = item
                elif isinstance(data, dict):
                    # Handle different possible structures
                    items = data.get("images", data.get("data", data.get("metadata", [])))
                    for item in items:
                        filename = item.get("filename")
                        if filename:
                            self.dataset_metadata[filename] = item
                
                logger.info(f"📋 Dataset metadata loaded: {len(self.dataset_metadata)} entries")
            except Exception as e:
                logger.warning(f"⚠️ Could not load dataset metadata: {e}")
                self.dataset_metadata = {}
        else:
            logger.warning(f"⚠️ Dataset metadata not found: {dataset_metadata_path}")
            self.dataset_metadata = {}
    
    def get_true_caption_and_rank(self, filename: str) -> tuple[Optional[str], Optional[int]]:
        """Get true caption and baseline rank for a given filename."""
        if filename in self.dataset_metadata:
            item = self.dataset_metadata[filename]
            true_caption = item.get("caption", item.get("true_caption"))
            baseline_rank = item.get("baseline_rank")
            return true_caption, baseline_rank
        return None, None
    
    def _encode_query(self, query_text: str) -> np.ndarray:
        """Encode text query into CLIP embedding."""
        with torch.no_grad():
            inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                feats = self.clip_model.get_text_features(**inputs)
        feats = feats.float().cpu().numpy()
        if self.index_metric == "ip":
            faiss.normalize_L2(feats)
        return feats
    
    def _generate_blip_caption(self, image_path: Path) -> str:
        """Generate BLIP caption for a single image with precise parameters."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    # Use precise parameters for detailed descriptions
                    out = self.blip_model.generate(
                        **inputs,
                        max_length=150,        # Longer descriptions
                        min_length=30,         # Ensure minimum detail
                        num_beams=8,           # Better search
                        no_repeat_ngram_size=3, # Avoid repetition
                        length_penalty=1.0,    # Balanced length
                        early_stopping=True,
                        do_sample=False        # Deterministic for consistency
                    )
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True).strip()
                return caption
        except Exception as e:
            logger.warning(f"Failed to generate BLIP caption for {image_path}: {e}")
            return "Could not generate caption"
    
    def _add_blip_captions(self, results: List[Dict]) -> List[Dict]:
        """Add BLIP-generated captions to retrieval results."""
        for result in results:
            image_path = self.images_dir / result['filename']
            blip_caption = self._generate_blip_caption(image_path)
            result['blip_caption'] = blip_caption
        return results
    
    def _call_ollama_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call Ollama LLM to reformulate query with automatic retry on failure."""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": float(self.llm_temp),
                        "top_p": float(self.llm_top_p)
                    }
                }
                
                cmd = [
                    "curl", "-s", "-X", "POST",
                    "http://localhost:11434/api/generate",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(payload)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.llm_timeout)
                if result.returncode != 0:
                    raise RuntimeError(f"Curl failed: {result.stderr}")
                
                response = json.loads(result.stdout)
                return response.get("response", "").strip()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    import time
                    time.sleep(2)  # Wait 2 seconds before retry
                else:
                    logger.error(f"❌ LLM call failed after {max_retries} attempts: {e}")
                    return ""
        
        return ""
    
    def _extract_features_from_query(self, query_text: str) -> Dict[str, str]:
        """Extract scene, emotion, and object features from a text query using LLM."""
        prompt = f"""Parse this image description into 3 specific features:

DESCRIPTION: "{query_text}"

Extract and return ONLY the following 3 features in this exact format:
SCENE: [describe the setting/location/environment]
EMOTION: [describe the mood/feeling/atmosphere]
OBJECT: [list the main objects/subjects/people]

Be concise and specific for each feature."""

        response = self._call_ollama_llm(prompt)
        
        # Parse the response to extract features with more robust parsing
        features = {"scene": "", "emotion": "", "object": ""}
        
        if response:
            logger.debug(f"🔍 Feature extraction response: {response}")
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # More flexible parsing - handle variations
                line_upper = line.upper()
                if line_upper.startswith("SCENE:"):
                    features["scene"] = line.split(":", 1)[1].strip() if ":" in line else ""
                elif line_upper.startswith("EMOTION:"):
                    features["emotion"] = line.split(":", 1)[1].strip() if ":" in line else ""
                elif line_upper.startswith("OBJECT:") or line_upper.startswith("OBJECTS:"):
                    features["object"] = line.split(":", 1)[1].strip() if ":" in line else ""
            logger.debug(f"✅ Parsed features: {features}")
        else:
            logger.warning("⚠️ No LLM response for feature extraction from query")
        
        return features
    
    def _extract_features_from_blip_descriptions(self, top3_results: List[Dict]) -> List[Dict[str, str]]:
        """Extract features from BLIP descriptions of top 3 images."""
        features_list = []
        
        for i, result in enumerate(top3_results, 1):
            blip_description = result.get('blip_caption', '')
            
            prompt = f"""Parse this image description into 3 specific features:

DESCRIPTION: "{blip_description}"

Extract and return ONLY the following 3 features in this exact format:
SCENE: [describe the setting/location/environment]
EMOTION: [describe the mood/feeling/atmosphere]
OBJECT: [list the main objects/subjects/people]

Be concise and specific for each feature."""

            response = self._call_ollama_llm(prompt)
            
            # Parse the response to extract features with robust parsing
            features = {"scene": "", "emotion": "", "object": ""}
            
            if response:
                logger.debug(f"🔍 LLM response for BLIP feature extraction {i}: {response}")
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # More flexible parsing - handle variations and case insensitivity
                    line_upper = line.upper()
                    if line_upper.startswith("SCENE:"):
                        features["scene"] = line.split(":", 1)[1].strip() if ":" in line else ""
                    elif line_upper.startswith("EMOTION:"):
                        features["emotion"] = line.split(":", 1)[1].strip() if ":" in line else ""
                    elif line_upper.startswith("OBJECT:") or line_upper.startswith("OBJECTS:"):
                        features["object"] = line.split(":", 1)[1].strip() if ":" in line else ""
                logger.debug(f"✅ Extracted features for image {i}: {features}")
            else:
                logger.warning(f"⚠️ No LLM response for BLIP feature extraction {i}")
            
            features_list.append(features)
        
        return features_list
    
    def _score_feature_alignment(self, query_features: Dict[str, str], image_features_list: List[Dict[str, str]]) -> List[float]:
        """Score global alignment between query features and each image's features."""
        alignment_scores = []
        
        for i, image_features in enumerate(image_features_list, 1):
            prompt = f"""Score the overall alignment between the query features and image features on a scale of 0.0 to 1.0.

QUERY FEATURES:
- SCENE: {query_features['scene']}
- EMOTION: {query_features['emotion']}
- OBJECT: {query_features['object']}

IMAGE {i} FEATURES:
- SCENE: {image_features['scene']}
- EMOTION: {image_features['emotion']}
- OBJECT: {image_features['object']}

Consider how well all three features (scene, emotion, object) align together as a whole.
Rate the overall alignment as a single global score (0.0 = no match, 1.0 = perfect match):

GLOBAL_SCORE: [0.0-1.0]

Return only the score in the exact format above."""

            response = self._call_ollama_llm(prompt)
            
            # Parse global score with robust multi-strategy parsing
            global_score = 0.0
            
            if response:
                logger.debug(f"🔍 LLM response for image {i} alignment: {response}")
                lines = response.strip().split('\n')
                
                # Strategy 1: Look for exact format "GLOBAL_SCORE: X.X"
                for line in lines:
                    line = line.strip()
                    line_upper = line.upper()
                    if "GLOBAL_SCORE" in line_upper or "GLOBAL SCORE" in line_upper:
                        try:
                            # Extract number after colon
                            if ":" in line:
                                score_part = line.split(":", 1)[1].strip()
                                # Remove any brackets or extra text
                                score_part = score_part.replace("[", "").replace("]", "").strip()
                                # Extract first number found
                                import re
                                numbers = re.findall(r'\d+\.?\d*', score_part)
                                if numbers:
                                    global_score = float(numbers[0])
                                    logger.debug(f"✅ Parsed score (strategy 1) for image {i}: {global_score}")
                                    break
                        except (ValueError, IndexError) as e:
                            logger.debug(f"⚠️ Strategy 1 failed: {e}")
                            continue
                
                # Strategy 2: Look for any decimal number between 0 and 1 in the response
                if global_score == 0.0:
                    import re
                    # Find all decimal numbers in response
                    numbers = re.findall(r'\b0\.\d+\b|\b1\.0+\b|\b[01]\b', response)
                    if numbers:
                        try:
                            global_score = float(numbers[0])
                            logger.debug(f"✅ Parsed score (strategy 2) for image {i}: {global_score}")
                        except ValueError:
                            pass
                
                # Strategy 3: Look for percentage and convert to 0-1 scale
                if global_score == 0.0:
                    import re
                    percentages = re.findall(r'(\d+)%', response)
                    if percentages:
                        try:
                            global_score = float(percentages[0]) / 100.0
                            logger.debug(f"✅ Parsed score from percentage (strategy 3) for image {i}: {global_score}")
                        except ValueError:
                            pass
                
                if global_score == 0.0:
                    logger.warning(f"⚠️ Could not parse alignment score for image {i} from response: {response[:100]}...")
            else:
                logger.warning(f"⚠️ No LLM response for image {i} alignment scoring")
            
            alignment_scores.append(global_score)
        
        return alignment_scores
    
    def _reformulate_query_with_coverage_improvement(self, original_query: str, query_features: Dict[str, str], 
                                                   alignment_scores: List[float], 
                                                   image_features_list: List[Dict[str, str]],
                                                   stage1_results: List[Dict]) -> str:
        """Reformulate query by focusing on misaligned elements from the two least aligned images.
        
        When alignment scores are tied, uses CLIP similarity score as tie-breaker (lower is worse).
        """
        
        # Find the two images with the lowest alignment scores
        # Include CLIP similarity scores for tie-breaking
        scored_images = []
        for i, (align_score, features, result) in enumerate(zip(alignment_scores, image_features_list, stage1_results), 1):
            clip_score = result.get('score', 0.0)  # CLIP similarity score
            scored_images.append((align_score, clip_score, features, i))
        
        # Sort by alignment score (lowest first), then by CLIP score (lowest first) for tie-breaking
        scored_images.sort(key=lambda x: (x[0], x[1]))
        
        # Take the two least aligned images
        least_aligned = scored_images[:2]
        
        # Log the selected images for reformulation
        logger.debug(f"🎯 Selected least aligned images for reformulation:")
        for align_score, clip_score, features, img_num in least_aligned:
            logger.debug(f"   Image {img_num}: alignment={align_score:.2f}, CLIP={clip_score:.4f}")
        
        # Build context for the two least aligned images
        misaligned_context = []
        for align_score, clip_score, features, img_num in least_aligned:
            misaligned_context.append(f"Image {img_num} (alignment: {align_score:.2f}, CLIP similarity: {clip_score:.4f}):")
            misaligned_context.append(f"  - Scene: {features['scene']}")
            misaligned_context.append(f"  - Emotion: {features['emotion']}")
            misaligned_context.append(f"  - Object: {features['object']}")
        
        misaligned_text = "\n".join(misaligned_context)
        
        prompt = f"""You are a query reformulation expert for image retrieval.

ORIGINAL QUERY: "{original_query}"

ORIGINAL QUERY FEATURES:
- SCENE: {query_features['scene']}
- EMOTION: {query_features['emotion']}
- OBJECT: {query_features['object']}

TWO LEAST ALIGNED RETRIEVED IMAGES:
{misaligned_text}

TASK:
Analyze the two least aligned retrieved images, why they misaligned and detect:
1. Which elements are drifting or inaccurate compared to the original query
2. Which crucial elements from the original query are missing or poorly represented

Then reformulate the query to:
- INSIST on the crucial elements that were missed or misrepresented
- CLARIFY ambiguous aspects that led to the wrong results
- STRENGTHEN the core intent of the original query

Return ONLY the reformulated query as a single line."""

        response = self._call_ollama_llm(prompt)
        
        if not response:
            logger.warning("⚠️ Reformulation failed, using original query")
            return original_query
        
        # Extract the reformulated query
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        return lines[0] if lines else original_query
    
    def _find_target_image_rank(self, query_text: str, target_filename: str) -> Optional[int]:
        """Find the rank of the target image in the retrieval results by searching progressively."""
        try:
            feats = self._encode_query(query_text)
            total_vectors = self.faiss_index.ntotal
            logger.info(f"🔍 Searching for {target_filename} in {total_vectors} vectors...")
            
            # Search progressively in batches
            batch_sizes = [100, 500, 1000, 5000, total_vectors]
            
            for batch_size in batch_sizes:
                search_size = min(batch_size, total_vectors)
                logger.debug(f"Searching in top {search_size} results...")
                
                raw_vals, ids = self.faiss_index.search(feats, search_size)
                raw_vals, ids = raw_vals[0], ids[0]
                
                for rank, rid in enumerate(ids, 1):
                    if rid < 0:  # Skip invalid IDs
                        continue
                        
                    eid = int(rid)
                    meta = self._id_to_meta.get(eid)
                    
                    # Try direct lookup first, then fallback to index-based lookup
                    if not meta and not self.uses_idmap and 0 <= eid < len(self.metadata):
                        meta = self.metadata[eid]
                    
                    if meta:
                        filename = meta.get("filename", "")
                        if filename == target_filename:
                            logger.info(f"✅ Found {target_filename} at rank {rank}")
                            return rank
                
                # If found in this batch, we would have returned already
                if search_size >= total_vectors:
                    break
            
            logger.warning(f"❌ Target image {target_filename} not found in any results")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to find target image rank: {e}")
            return None
    
    def _debug_target_image(self, target_filename: str):
        """Debug function to check if target image exists in metadata."""
        logger.info(f"🔍 Debugging target image: {target_filename}")
        
        # Check in dataset metadata
        if target_filename in self.dataset_metadata:
            logger.info(f"✅ Found in dataset metadata")
        else:
            logger.warning(f"❌ Not found in dataset metadata")
        
        # Check in vector metadata
        found_in_vector_meta = False
        for i, meta in enumerate(self.metadata):
            if meta.get("filename") == target_filename:
                logger.info(f"✅ Found in vector metadata at index {i}, embedding_id: {meta.get('embedding_id', i)}")
                found_in_vector_meta = True
                break
        
        if not found_in_vector_meta:
            logger.warning(f"❌ Not found in vector metadata")
            # Show first few filenames for comparison
            sample_files = [m.get("filename", "no_filename") for m in self.metadata[:5]]
            logger.info(f"Sample filenames in metadata: {sample_files}")
        
        # Check in id_to_meta mapping
        found_in_mapping = False
        for eid, meta in self._id_to_meta.items():
            if meta.get("filename") == target_filename:
                logger.info(f"✅ Found in id_to_meta mapping at embedding_id: {eid}")
                found_in_mapping = True
                break
        
        if not found_in_mapping:
            logger.warning(f"❌ Not found in id_to_meta mapping")
    
    def retrieve_top3(self, query_text: str) -> List[Dict]:
        """
        Retrieve top 3 images from a text prompt.
        
        Args:
            query_text: The textual query/prompt
            
        Returns:
            List of 3 dictionaries containing:
                - rank: Rank position (1-3)
                - faiss_index: FAISS index ID
                - score: Standardized similarity score (higher = better)
                - raw_faiss_value: Raw FAISS distance/similarity
                - filename: Image filename
                - blip_caption: BLIP-generated caption
        """
        results = self.retrieve_topk(query_text, k=3)
        return self._add_blip_captions(results)
    
    def two_stage_retrieve(self, query_text: str, target_filename: Optional[str] = None) -> Dict:
        """
        Two-stage retrieval with structured feature analysis and coverage-based reformulation.
        
        Args:
            query_text: Original textual query
            target_filename: Filename of the target image to track its rank
            
        Returns:
            Dictionary containing:
                - original_query: Original query text
                - query_features: Extracted features from original query
                - stage1_top3: Top 3 results from first retrieval with BLIP captions
                - image_features: Extracted features from top 3 BLIP descriptions
                - alignment_scores: Feature alignment scores for each image
                - reformulated_query: Coverage-improved reformulated query
                - stage2_top3: Top 3 final results from reformulated query with BLIP captions
                - target_rank_after_reformulation: New rank of target image (if provided)
        """
        # Step 1: Extract features from original query
        logger.info("🔍 Step 1: Extracting features from original query...")
        query_features = self._extract_features_from_query(query_text)
        logger.info(f"📋 Query features - Scene: {query_features['scene']}, Emotion: {query_features['emotion']}, Object: {query_features['object']}")
        
        # Step 2: Initial retrieval and BLIP captioning
        logger.info("🔍 Step 2: Initial retrieval and BLIP captioning...")
        stage1_top3 = self.retrieve_top3(query_text)
        
        # Step 3: Extract features from BLIP descriptions
        logger.info("🔍 Step 3: Extracting features from BLIP descriptions...")
        image_features = self._extract_features_from_blip_descriptions(stage1_top3)
        
        # Step 4: Score feature alignment
        logger.info("🔍 Step 4: Scoring feature alignment...")
        alignment_scores = self._score_feature_alignment(query_features, image_features)
        
        # Log alignment analysis
        for i, score in enumerate(alignment_scores, 1):
            logger.info(f"🎯 Image {i} global alignment score: {score:.2f}")
        
        # Step 5: Coverage-based query reformulation
        logger.info("🤖 Step 5: Coverage-based query reformulation...")
        reformulated_query = self._reformulate_query_with_coverage_improvement(
            query_text, query_features, alignment_scores, image_features, stage1_top3
        )
        logger.info(f"✨ Reformulated query: '{reformulated_query}'")
        
        # Step 6: Re-retrieval with reformulated query
        logger.info(f"🔍 Step 6: Re-retrieval with reformulated query...")
        stage2_top3 = self.retrieve_top3(reformulated_query)
        
        # Step 7: Find new rank of target image if provided
        target_rank_after = None
        if target_filename:
            logger.info(f"🎯 Step 7: Finding new rank of target image: {target_filename}")
            target_rank_after = self._find_target_image_rank(reformulated_query, target_filename)
            
            # If not found, run debug
            if target_rank_after is None:
                self._debug_target_image(target_filename)
        
        return {
            "original_query": query_text,
            "query_features": query_features,
            "stage1_top3": stage1_top3,
            "image_features": image_features,
            "alignment_scores": alignment_scores,
            "reformulated_query": reformulated_query,
            "stage2_top3": stage2_top3,
            "target_rank_after_reformulation": target_rank_after
        }
    
    def retrieve_topk(self, query_text: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top-k images from a text prompt.
        
        Args:
            query_text: The textual query/prompt
            k: Number of top results to return
            
        Returns:
            List of k dictionaries containing retrieval results
        """
        feats = self._encode_query(query_text)
        raw_vals, ids = self.faiss_index.search(feats, k)
        raw_vals, ids = raw_vals[0], ids[0]
        std_scores = _standardize_scores(raw_vals, self.index_metric)
        
        results = []
        for rank, (std, raw, rid) in enumerate(zip(std_scores, raw_vals, ids), 1):
            eid = int(rid)
            meta = self._id_to_meta.get(eid)
            if not meta and not self.uses_idmap and 0 <= eid < len(self.metadata):
                meta = self.metadata[eid]
            if not meta:
                logger.warning(f"❌ No metadata found for id {eid}")
                continue
            results.append({
                "rank": rank,
                "faiss_index": eid,
                "score": float(std),            # standardized, higher = better
                "raw_faiss_value": float(raw), # raw FAISS value
                "filename": meta.get("filename", f"image_{eid}.jpg")
            })
        return results
    
    def display_results(self, query_text: str, results: List[Dict], true_caption: Optional[str] = None, baseline_rank: Optional[int] = None):
        """Display retrieval results in a formatted way."""
        print("\n" + "=" * 80)
        print("🔍 IMAGE RETRIEVAL RESULTS")
        print("=" * 80)
        print(f"📝 Query: {query_text}")
        
        if true_caption:
            print(f"✅ True Caption: {true_caption}")
        if baseline_rank is not None:
            print(f"📊 Baseline Rank: {baseline_rank}")
        
        print(f"\n🎯 Top {len(results)} Results:")
        for item in results:
            print(f"\n  Rank #{item['rank']}")
            print(f"  📄 File: {item['filename']}")
            print(f"  📊 Similarity Score: {item['score']:.4f}")
            if 'blip_caption' in item:
                print(f"  🤖 BLIP Caption: {item['blip_caption']}")
        print("=" * 80 + "\n")
    
    def display_two_stage_results(self, result: Dict, true_caption: Optional[str] = None, baseline_rank: Optional[int] = None, target_filename: Optional[str] = None):
        """Display structured two-stage retrieval results with feature analysis."""
        print("\n" + "=" * 80)
        print("🔁 STRUCTURED TWO-STAGE RETRIEVAL WITH FEATURE ANALYSIS")
        print("=" * 80)
        print(f"📝 Original Query: {result['original_query']}")
        
        if true_caption:
            print(f"✅ True Caption: {true_caption}")
        if baseline_rank is not None:
            print(f"📊 Baseline Rank: {baseline_rank}")
        print(f"🤖 LLM Model: {self.llm_model}")
        
        # Show rank progression if we have both baseline and new rank
        new_rank = result.get("target_rank_after_reformulation")
        if baseline_rank is not None and new_rank is not None:
            if new_rank < baseline_rank:
                improvement = baseline_rank - new_rank
                print(f"📈 Rank Improvement: {baseline_rank} → {new_rank} (↑{improvement} positions)")
            elif new_rank > baseline_rank:
                degradation = new_rank - baseline_rank
                print(f"📉 Rank Degradation: {baseline_rank} → {new_rank} (↓{degradation} positions)")
            else:
                print(f"🔄 Rank Unchanged: {baseline_rank} → {new_rank}")
        elif new_rank is not None:
            print(f"🎯 New Rank After Reformulation: {new_rank}")
        
        # Display extracted query features
        print(f"\n� Extracted Query Features:")
        query_features = result.get("query_features", {})
        print(f"  🏞️  Scene: {query_features.get('scene', 'N/A')}")
        print(f"  😊 Emotion: {query_features.get('emotion', 'N/A')}")
        print(f"  🎯 Object: {query_features.get('object', 'N/A')}")
        
        print("\n�🔍 Stage 1 — Top 3 from initial retrieval:")
        alignment_scores = result.get("alignment_scores", [])
        image_features = result.get("image_features", [])
        
        for i, item in enumerate(result["stage1_top3"]):
            print(f"  #{item['rank']:>1}  📄 {item['filename']}  📊 {item['score']:.4f}")
            print(f"      🤖 BLIP: {item.get('blip_caption', 'N/A')}")
            
            # Show extracted features and alignment scores
            if i < len(image_features):
                features = image_features[i]
                print(f"      📋 Features - Scene: {features.get('scene', 'N/A')}")
                print(f"                   Emotion: {features.get('emotion', 'N/A')}")
                print(f"                   Object: {features.get('object', 'N/A')}")
            
            if i < len(alignment_scores):
                score = alignment_scores[i]
                print(f"      📊 Global Alignment Score: {score:.2f}")
        
        print(f"\n✨ Coverage-Improved Reformulated Query:")
        print(f"  {result['reformulated_query']}")
        
        print(f"\n🔍 Stage 2 — Top 3 from reformulated query:")
        for item in result["stage2_top3"]:
            print(f"  #{item['rank']:>1}  📄 {item['filename']}  📊 {item['score']:.4f}")
            print(f"      🤖 BLIP: {item.get('blip_caption', 'N/A')}")
        
        print("=" * 80 + "\n")

# --------------------------------- CLI -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Retrieve top images from a text prompt")
    parser.add_argument('--vectordb', required=True, help='VectorDB name, e.g., COCO_VectorDB')
    parser.add_argument('--vectordb-dir', default='VectorDBs', help='Directory with .index and *_metadata.json')
    parser.add_argument('--prompt', type=str, help='Text prompt for image retrieval')
    parser.add_argument('--filename', type=str, help='Use true caption of this filename as query')
    parser.add_argument('--output', type=str, help='Optional: Save results to JSON file')
    # LLM parameters
    parser.add_argument('--llm', help='LLM model for reformulation (default: gemma3:4b)')
    parser.add_argument('--llm-temp', type=float, help='LLM temperature (default: 0.1)')
    parser.add_argument('--llm-top-p', type=float, help='LLM top_p (default: 0.9)')
    parser.add_argument('--llm-timeout', type=int, help='LLM timeout in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Create retriever
    retriever = ImageRetriever(
        vectordb_name=args.vectordb,
        vectordb_dir=args.vectordb_dir,
    )
    
    # Determine query text and get true caption/baseline rank if filename provided
    true_caption = None
    baseline_rank = None
    
    if args.filename:
        true_caption, baseline_rank = retriever.get_true_caption_and_rank(args.filename)
        if true_caption:
            query_text = true_caption
            logger.info(f"📄 Using true caption for {args.filename}")
        else:
            logger.error(f"❌ Could not find true caption for filename: {args.filename}")
            sys.exit(1)
    elif args.prompt:
        query_text = args.prompt
    else:
        logger.error("❌ Please provide either --prompt 'your text here' or --filename 'image.jpg'")
        sys.exit(1)
    
    # Always use two-stage retrieval with LLM reformulation
    logger.info(f"🔍 Two-stage retrieval for: '{query_text}'")
    result = retriever.two_stage_retrieve(query_text, target_filename=args.filename)
    retriever.display_two_stage_results(result, true_caption, baseline_rank, target_filename=args.filename)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "vectordb": args.vectordb,
            "filename": args.filename,
            "true_caption": true_caption,
            "baseline_rank": baseline_rank,
            "llm_model": args.llm,
            **result
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()
