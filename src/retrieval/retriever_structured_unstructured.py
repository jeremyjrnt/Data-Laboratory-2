#!/usr/bin/env python3
"""
Combined Structured + Unstructured Image Retrieval using CLIP and FAISS.
Compares both unstructured and structured reformulation approaches on the same image.

python src\retrieval\retriever_structured_unstructured.py --vectordb COCO_VectorDB --filename 000000000077.jpg

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
logger = logging.getLogger("image_retrieval_combined")
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
class CombinedImageRetriever:
    """
    Combined image retrieval system using both unstructured and structured approaches.
    
    Args:
        vectordb_name: Name of the vector database (e.g., 'COCO_VectorDB')
        vectordb_dir: Directory containing the .index and _metadata.json files
        llm_model: LLM model for query reformulation
        llm_temp: LLM temperature
        llm_top_p: LLM top_p parameter
        llm_timeout: LLM timeout in seconds
    """
    
    def __init__(
        self,
        vectordb_name: str,
        vectordb_dir: str = "VectorDBs",
        llm_model: str = "gemma3:4b",
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
        # Extract just the filename if a full path is provided
        from pathlib import Path
        basename = Path(filename).name
        
        # Try both the full path and just the basename
        for key in [filename, basename]:
            if key in self.dataset_metadata:
                item = self.dataset_metadata[key]
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
        logger.info("🖼️ Generating BLIP descriptions...")
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
                    "http://100.64.0.7:11434/api/generate",
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
    
    # ==================== UNSTRUCTURED METHODS ====================
    
    def _build_reformulation_prompt_unstructured(self, original_query: str, top3_results: List[Dict]) -> str:
        """Build unstructured prompt for query reformulation based on BLIP captions."""
        blip_descriptions = []
        for i, result in enumerate(top3_results, 1):
            blip_descriptions.append(f"{i}. {result.get('blip_caption', 'No description')}")
        
        descriptions_text = "\n".join(blip_descriptions)
        
        prompt = f"""You are a query refinement assistant for image retrieval.

ORIGINAL QUERY:
"{original_query}"

TOP 3 IMAGE DESCRIPTIONS:
{descriptions_text}

TASK:
Compare the query with the image descriptions. Find what key details are missing or vague, and rewrite the query to:
- Add missing or underrepresented details
- Clarify any ambiguity
- Keep the same intent and style

Return only the improved query as one single line."""

        return prompt
    
    def reformulate_query_unstructured(self, original_query: str, top3_results: List[Dict]) -> str:
        """Reformulate query using unstructured LLM approach."""
        prompt = self._build_reformulation_prompt_unstructured(original_query, top3_results)
        reformulated = self._call_ollama_llm(prompt)
        
        if not reformulated:
            logger.warning("⚠️ Unstructured reformulation failed, using original query")
            return original_query
        
        # Extract just the reformulated query (remove any extra text)
        lines = [line.strip() for line in reformulated.strip().split('\n') if line.strip()]
        return lines[0] if lines else original_query
    
    def _ask_reformulation_decision_unstructured(self, original_query: str, top3_results: List[Dict]) -> bool:
        """Ask LLM if query reformulation is needed based on unstructured comparison."""
        
        blip_descriptions = []
        for i, result in enumerate(top3_results, 1):
            clip_score = result.get('score', 0.0)
            blip_descriptions.append(f"{i}. (CLIP score: {clip_score:.4f}) {result.get('blip_caption', 'No description')}")
        
        descriptions_text = "\n".join(blip_descriptions)
        
        prompt = f"""You are an expert in image retrieval quality assessment.

ORIGINAL QUERY:
"{original_query}"

TOP 3 IMAGE DESCRIPTIONS:
{descriptions_text}

TASK:
Analyze whether the retrieved images accurately satisfy the informational need expressed by the query,
not just its literal wording.

Consider:
1. Are the key details and concepts from the query clearly represented in the retrieved images' descriptions?
2. Are there important mismatches, missing elements, or irrelevant content?
3. Are there discrepancies between the informational need (what the query truly intends to express)
   and the informational formulation (how the query is written)?
4. Would reformulating the query likely improve alignment between the query and the retrieved images?

DECISION:
Based on your analysis, do you think query reformulation would improve the retrieval results?

Answer with ONLY one word: YES or NO"""

        response = self._call_ollama_llm(prompt)
        
        if not response:
            logger.warning("⚠️ LLM unstructured reformulation decision failed - Answer error")
            return None
        
        # Parse response - look for YES or NO
        response_upper = response.strip().upper()
        
        # Simple parsing - look for YES or NO in the response
        if "YES" in response_upper:
            return True
        elif "NO" in response_upper:
            return False
        else:
            logger.warning(f"⚠️ Could not parse LLM unstructured decision from: {response[:100]}... - Answer error")
            return None
    
    # ==================== STRUCTURED METHODS ====================
    
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
    
    def _ask_reformulation_decision(self, query_features: Dict[str, str], 
                                    image_features_list: List[Dict[str, str]],
                                    stage1_results: List[Dict]) -> bool:
        """Ask LLM if query reformulation is needed based on feature comparison."""
        
        # Build context with features of all 3 images
        images_features_context = []
        for i, (features, result) in enumerate(zip(image_features_list, stage1_results), 1):
            clip_score = result.get('score', 0.0)
            images_features_context.append(f"Image {i} (CLIP score: {clip_score:.4f}):")
            images_features_context.append(f"  - Scene: {features['scene']}")
            images_features_context.append(f"  - Emotion: {features['emotion']}")
            images_features_context.append(f"  - Object: {features['object']}")
        
        images_text = "\n".join(images_features_context)
        
        prompt = f"""You are an expert in image retrieval quality assessment.

ORIGINAL QUERY FEATURES:
- Scene: {query_features['scene']}
- Emotion: {query_features['emotion']}
- Object: {query_features['object']}

TOP 3 RETRIEVED IMAGES FEATURES:
{images_text}

TASK:
Evaluate how well the retrieved images satisfy the informational need expressed by the query,
not just its literal wording.

Consider:
1. Do the scene, emotion, and object features accurately reflect the intended meaning of the query?
2. Are there discrepancies between the informational need (what the user truly seeks)
   and the informational formulation (how the query was written)?
3. Are there significant mismatches, omissions, or irrelevant elements among the retrieved images?

DECISION:
Based on your analysis, decide whether query reformulation could significantly improve retrieval quality.

Answer with ONLY one word: YES or NO"""

        response = self._call_ollama_llm(prompt)
        
        if not response:
            logger.warning("⚠️ LLM reformulation decision failed - Answer error")
            return None
        
        # Parse response - look for YES or NO
        response_upper = response.strip().upper()
        
        # Simple parsing - look for YES or NO in the response
        if "YES" in response_upper:
            return True
        elif "NO" in response_upper:
            return False
        else:
            logger.warning(f"⚠️ Could not parse LLM decision from: {response[:100]}... - Answer error")
            return None
    
    def _reformulate_query_structured(self, original_query: str, query_features: Dict[str, str], 
                                     image_features_list: List[Dict[str, str]],
                                     stage1_results: List[Dict]) -> str:
        """Reformulate query using features from all 3 retrieved images."""
        
        # Build context with features of all 3 images
        images_features_context = []
        for i, (features, result) in enumerate(zip(image_features_list, stage1_results), 1):
            clip_score = result.get('score', 0.0)
            images_features_context.append(f"Image {i} Features (CLIP score: {clip_score:.4f}):")
            images_features_context.append(f"  - Scene: {features['scene']}")
            images_features_context.append(f"  - Emotion: {features['emotion']}")
            images_features_context.append(f"  - Object: {features['object']}")
        
        images_text = "\n".join(images_features_context)
        
        prompt = f"""You are an expert in query reformulation for image retrieval.

ORIGINAL QUERY:
"{original_query}"

QUERY FEATURES:
- Scene: {query_features['scene']}
- Emotion: {query_features['emotion']}
- Object: {query_features['object']}

TOP 3 RETRIEVED IMAGES (with their extracted features):
{images_text}

TASK:
Compare the original query features with the three retrieved images' features.

1. Identify which query features (scene, emotion, object) are poorly matched or absent in those images.
2. Determine what visual or emotional aspects of the original query need to be reinforced or clarified.
3. Reformulate the query to:
   - Emphasize underrepresented or missing features
   - Clarify ambiguous terms or contexts that caused mismatches
   - Preserve the original intent and core visual meaning

OUTPUT:
Return ONLY the improved reformulated query as one clear, natural sentence. Do not explain or justify your reasoning."""

        response = self._call_ollama_llm(prompt)
        
        if not response:
            logger.warning("⚠️ Structured reformulation failed, using original query")
            return original_query
        
        # Extract the reformulated query
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        return lines[0] if lines else original_query
    
    # ==================== RANK FINDING ====================
    
    def _find_target_image_rank(self, query_text: str, target_filename: str) -> Optional[int]:
        """Find the rank of the target image in the retrieval results by searching progressively."""
        try:
            # Extract basename from target filename for comparison
            from pathlib import Path
            target_basename = Path(target_filename).name
            
            feats = self._encode_query(query_text)
            total_vectors = self.faiss_index.ntotal
            logger.debug(f"🔍 Searching for {target_filename} (basename: {target_basename}) in {total_vectors} vectors...")
            
            # Search progressively in batches
            batch_sizes = [100, 500, 1000, 5000, total_vectors]
            
            for batch_size in batch_sizes:
                search_size = min(batch_size, total_vectors)
                
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
                        # Compare both full path and basename
                        if filename == target_filename or filename == target_basename:
                            return rank
                
                # If found in this batch, we would have returned already
                if search_size >= total_vectors:
                    break
            
            logger.warning(f"❌ Target image {target_filename} not found in any results")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to find target image rank: {e}")
            return None
    
    # ==================== CORE RETRIEVAL ====================
    
    def retrieve_topk(self, query_text: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k images from a text prompt."""
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
    
    def retrieve_top3(self, query_text: str) -> List[Dict]:
        """Retrieve top 3 images from a text prompt."""
        results = self.retrieve_topk(query_text, k=3)
        return self._add_blip_captions(results)
    
    # ==================== COMBINED TWO-STAGE RETRIEVAL ====================
    
    def combined_two_stage_retrieve(self, query_text: str, target_filename: Optional[str] = None) -> Dict:
        """
        Combined two-stage retrieval: first unstructured, then structured.
        BLIP captions are generated once and reused for both methods.
        
        Args:
            query_text: Original textual query
            target_filename: Filename of the target image to track its rank
            
        Returns:
            Dictionary containing results from both methods
        """
        logger.info("\n" + "="*80)
        logger.info("🔁 COMBINED STRUCTURED + UNSTRUCTURED RETRIEVAL")
        logger.info("="*80)
        
        # Get baseline rank from metadata if target is provided
        baseline_rank = None
        if target_filename:
            # Retrieve baseline_rank from dataset metadata (already calculated)
            _, baseline_rank = self.get_true_caption_and_rank(target_filename)
            if baseline_rank:
                logger.info(f"📊 Baseline rank for {target_filename}: {baseline_rank}")
            else:
                logger.warning(f"⚠️ Baseline rank not found for {target_filename}")
        
        # Step 1: Initial retrieval and BLIP captioning (shared by both methods)
        logger.info("\n🔍 Step 1: Initial retrieval and BLIP captioning (shared)...")
        stage1_top3 = self.retrieve_top3(query_text)
        
        # ==================== UNSTRUCTURED METHOD ====================
        logger.info("\n" + "="*80)
        logger.info("📝 UNSTRUCTURED METHOD")
        logger.info("="*80)
        
        # Step 2a: LLM decision on whether reformulation is needed (unstructured)
        logger.info("🤖 Step 2a: Asking LLM if query reformulation is needed (unstructured)...")
        unstructured_reformulation_decision = self._ask_reformulation_decision_unstructured(query_text, stage1_top3)
        if unstructured_reformulation_decision is True:
            logger.info(f"💡 LLM decision (unstructured): YES - reformulation recommended")
        elif unstructured_reformulation_decision is False:
            logger.info(f"💡 LLM decision (unstructured): NO - reformulation not needed")
        else:
            logger.info(f"💡 LLM decision (unstructured): ANSWER ERROR - could not determine")
        
        # Step 3a: Unstructured query reformulation
        logger.info("🤖 Step 3a: Unstructured query reformulation...")
        unstructured_reformulated = self.reformulate_query_unstructured(query_text, stage1_top3)
        logger.info(f"✨ Unstructured reformulated query: '{unstructured_reformulated}'")
        
        # Step 4a: Re-retrieval with unstructured reformulated query
        logger.info("🔍 Step 4a: Re-retrieval with unstructured reformulated query...")
        unstructured_stage2_top3 = self.retrieve_top3(unstructured_reformulated)
        
        # Step 5a: Find new rank with unstructured method
        unstructured_new_rank = None
        if target_filename:
            # If reformulation failed (query unchanged), new rank = baseline rank
            if unstructured_reformulated == query_text and baseline_rank:
                unstructured_new_rank = baseline_rank
                logger.info(f"📊 Unstructured reformulation failed, new rank = baseline rank: {unstructured_new_rank}")
            else:
                logger.info(f"🎯 Step 5a: Finding new rank with unstructured method...")
                unstructured_new_rank = self._find_target_image_rank(unstructured_reformulated, target_filename)
                if unstructured_new_rank:
                    logger.info(f"📊 Unstructured new rank: {unstructured_new_rank}")
        
        # Calculate unstructured progression
        unstructured_progression = None
        if baseline_rank and unstructured_new_rank:
            unstructured_progression = baseline_rank - unstructured_new_rank
            logger.info(f"📈 Unstructured progression: {baseline_rank} → {unstructured_new_rank} ({'+' if unstructured_progression > 0 else ''}{unstructured_progression})")
        
        # Evaluate LLM reformulation decision (unstructured)
        unstructured_decision_correct = None
        if unstructured_reformulation_decision is not None and unstructured_progression is not None:
            # LLM said YES (reformulation needed)
            if unstructured_reformulation_decision:
                if unstructured_progression > 0:
                    unstructured_decision_correct = True
                    logger.info("✅ LLM decision (unstructured) CORRECT: Said YES and it improved")
                elif unstructured_progression < 0:
                    unstructured_decision_correct = False
                    logger.info("❌ LLM decision (unstructured) WRONG: Said YES but it degraded")
                else:  # progression == 0
                    unstructured_decision_correct = False
                    logger.info("❌ LLM decision (unstructured) WRONG: Said YES but no change (unnecessary reformulation)")
            # LLM said NO (reformulation not needed)
            else:
                if unstructured_progression > 0:
                    unstructured_decision_correct = False
                    logger.info("❌ LLM decision (unstructured) WRONG: Said NO but it would have improved")
                elif unstructured_progression < 0:
                    unstructured_decision_correct = True
                    logger.info("✅ LLM decision (unstructured) CORRECT: Said NO and reformulation degraded")
                else:  # progression == 0
                    unstructured_decision_correct = True
                    logger.info("✅ LLM decision (unstructured) CORRECT: Said NO and no change (reformulation was unnecessary)")
        elif unstructured_reformulation_decision is None:
            logger.info("⚠️  Cannot evaluate LLM decision (unstructured): Answer error")
            unstructured_decision_correct = None
        
        # ==================== STRUCTURED METHOD ====================
        logger.info("\n" + "="*80)
        logger.info("📊 STRUCTURED METHOD")
        logger.info("="*80)
        
        # Step 2b: Extract features from original query
        logger.info("🔍 Step 2b: Extracting features from original query...")
        query_features = self._extract_features_from_query(query_text)
        logger.info(f"📋 Query features - Scene: {query_features['scene']}, Emotion: {query_features['emotion']}, Object: {query_features['object']}")
        
        # Step 3b: Extract features from BLIP descriptions (reuse same BLIP captions)
        logger.info("🔍 Step 3b: Extracting features from BLIP descriptions...")
        image_features = self._extract_features_from_blip_descriptions(stage1_top3)

        # Step 4b: LLM decision on whether reformulation is needed
        logger.info("🤖 Step 4b: Asking LLM if query reformulation is needed...")
        reformulation_decision = self._ask_reformulation_decision(query_features, image_features, stage1_top3)
        if reformulation_decision is True:
            logger.info(f"💡 LLM decision: YES - reformulation recommended")
        elif reformulation_decision is False:
            logger.info(f"💡 LLM decision: NO - reformulation not needed")
        else:
            logger.info(f"💡 LLM decision: ANSWER ERROR - could not determine")

        # Step 5b: Structured query reformulation (using all 3 images' features)
        logger.info("🤖 Step 5b: Structured query reformulation...")
        structured_reformulated = self._reformulate_query_structured(
            query_text, query_features, image_features, stage1_top3
        )
        logger.info(f"✨ Structured reformulated query: '{structured_reformulated}'")
        
        # Step 6b: Re-retrieval with structured reformulated query
        logger.info("🔍 Step 6b: Re-retrieval with structured reformulated query...")
        structured_stage2_top3 = self.retrieve_top3(structured_reformulated)

        # Step 7b: Find new rank with structured method
        structured_new_rank = None
        if target_filename:
            # If reformulation failed (query unchanged), new rank = baseline rank
            if structured_reformulated == query_text:
                structured_new_rank = baseline_rank
                logger.info(f"📊 Structured reformulation failed, new rank = baseline rank: {structured_new_rank}")
            else:
                logger.info(f"🎯 Step 7b: Finding new rank with structured method...")
                structured_new_rank = self._find_target_image_rank(structured_reformulated, target_filename)
                if structured_new_rank:
                    logger.info(f"📊 Structured new rank: {structured_new_rank}")
        
        # Calculate structured progression
        structured_progression = None
        if baseline_rank and structured_new_rank:
            structured_progression = baseline_rank - structured_new_rank
            logger.info(f"📈 Structured progression: {baseline_rank} → {structured_new_rank} ({'+' if structured_progression > 0 else ''}{structured_progression})")
        
        # Evaluate LLM reformulation decision
        decision_correct = None
        if reformulation_decision is not None and structured_progression is not None:
            # LLM said YES (reformulation needed)
            if reformulation_decision:
                if structured_progression > 0:
                    decision_correct = True
                    logger.info("✅ LLM decision CORRECT: Said YES and it improved")
                elif structured_progression < 0:
                    decision_correct = False
                    logger.info("❌ LLM decision WRONG: Said YES but it degraded")
                else:  # progression == 0
                    decision_correct = False
                    logger.info("❌ LLM decision WRONG: Said YES but no change (unnecessary reformulation)")
            # LLM said NO (reformulation not needed)
            else:
                if structured_progression > 0:
                    decision_correct = False
                    logger.info("❌ LLM decision WRONG: Said NO but it would have improved")
                elif structured_progression < 0:
                    decision_correct = True
                    logger.info("✅ LLM decision CORRECT: Said NO and reformulation degraded")
                else:  # progression == 0
                    decision_correct = True
                    logger.info("✅ LLM decision CORRECT: Said NO and no change (reformulation was unnecessary)")
        elif reformulation_decision is None:
            logger.info("⚠️  Cannot evaluate LLM decision: Answer error")
            decision_correct = None
        
        return {
            "original_query": query_text,
            "baseline_rank": baseline_rank,
            "stage1_top3": stage1_top3,
            
            # Unstructured results
            "unstructured": {
                "llm_reformulation_decision": unstructured_reformulation_decision,
                "decision_correct": unstructured_decision_correct,
                "reformulated_query": unstructured_reformulated,
                "stage2_top3": unstructured_stage2_top3,
                "new_rank": unstructured_new_rank,
                "progression": unstructured_progression
            },
            
            # Structured results
            "structured": {
                "query_features": query_features,
                "image_features": image_features,
                "llm_reformulation_decision": reformulation_decision,
                "decision_correct": decision_correct,
                "reformulated_query": structured_reformulated,
                "stage2_top3": structured_stage2_top3,
                "new_rank": structured_new_rank,
                "progression": structured_progression
            }
        }
    
    # ==================== DISPLAY ====================
    
    def display_combined_results(self, result: Dict, true_caption: Optional[str] = None, target_filename: Optional[str] = None):
        """Display combined results from both unstructured and structured methods."""
        print("\n" + "=" * 80)
        print("🔁 COMBINED STRUCTURED + UNSTRUCTURED RETRIEVAL RESULTS")
        print("=" * 80)
        print(f"📝 Original Query: {result['original_query']}")
        
        if true_caption:
            print(f"✅ True Caption: {true_caption}")
        
        baseline_rank = result.get('baseline_rank')
        if baseline_rank is not None:
            print(f"📊 Baseline Rank: {baseline_rank}")
        
        print(f"🤖 LLM Model: {self.llm_model}")
        
        # Initial retrieval results
        print("\n🔍 Stage 1 — Initial Top 3:")
        for item in result["stage1_top3"]:
            print(f"  #{item['rank']:>1}  📄 {item['filename']}  📊 {item['score']:.4f}")
            print(f"      🤖 BLIP: {item.get('blip_caption', 'N/A')}")
        
        # ==================== UNSTRUCTURED RESULTS ====================
        print("\n" + "=" * 80)
        print("📝 UNSTRUCTURED METHOD RESULTS")
        print("=" * 80)
        
        unstructured = result["unstructured"]
        
        # Display LLM reformulation decision (unstructured)
        llm_decision_unstructured = unstructured.get("llm_reformulation_decision")
        decision_correct_unstructured = unstructured.get("decision_correct")
        
        print(f"\n🤖 LLM Reformulation Decision (Unstructured):")
        if llm_decision_unstructured is True:
            print(f"  Decision: YES (reformulation needed)")
        elif llm_decision_unstructured is False:
            print(f"  Decision: NO (reformulation not needed)")
        else:
            print(f"  Decision: ANSWER ERROR (could not determine)")
        
        if llm_decision_unstructured is not None:
            if decision_correct_unstructured is True:
                print(f"  Result: ✅ CORRECT")
            elif decision_correct_unstructured is False:
                print(f"  Result: ❌ WRONG")
        else:
            print(f"  Result: ⚠️  Cannot evaluate (answer error)")
        
        print(f"\n✨ Unstructured Reformulated Query:")
        print(f"  {unstructured['reformulated_query']}")
        
        # Show progression
        if unstructured['progression'] is not None:
            prog = unstructured['progression']
            if prog > 0:
                print(f"\n📈 Unstructured Rank Progression: {baseline_rank} → {unstructured['new_rank']} (↑{prog} positions)")
            elif prog < 0:
                print(f"\n📉 Unstructured Rank Degradation: {baseline_rank} → {unstructured['new_rank']} (↓{abs(prog)} positions)")
            else:
                print(f"\n🔄 Unstructured Rank Unchanged: {baseline_rank} → {unstructured['new_rank']}")
        elif unstructured['new_rank'] is not None:
            print(f"\n🎯 Unstructured New Rank: {unstructured['new_rank']}")
        
        print(f"\n🔍 Unstructured Stage 2 — Top 3:")
        for item in unstructured["stage2_top3"]:
            print(f"  #{item['rank']:>1}  📄 {item['filename']}  📊 {item['score']:.4f}")
            print(f"      🤖 BLIP: {item.get('blip_caption', 'N/A')}")
        
        # ==================== STRUCTURED RESULTS ====================
        print("\n" + "=" * 80)
        print("📊 STRUCTURED METHOD RESULTS")
        print("=" * 80)
        
        structured = result["structured"]
        
        print(f"\n📋 Extracted Query Features:")
        query_features = structured.get("query_features", {})
        print(f"  🏞️  Scene: {query_features.get('scene', 'N/A')}")
        print(f"  😊 Emotion: {query_features.get('emotion', 'N/A')}")
        print(f"  🎯 Object: {query_features.get('object', 'N/A')}")

        # Display LLM reformulation decision
        llm_decision = structured.get("llm_reformulation_decision")
        decision_correct = structured.get("decision_correct")
        
        print(f"\n🤖 LLM Reformulation Decision:")
        if llm_decision is True:
            print(f"  Decision: YES (reformulation needed)")
        elif llm_decision is False:
            print(f"  Decision: NO (reformulation not needed)")
        else:
            print(f"  Decision: ANSWER ERROR (could not determine)")
        
        if llm_decision is not None:
            if decision_correct is True:
                print(f"  Result: ✅ CORRECT")
            elif decision_correct is False:
                print(f"  Result: ❌ WRONG")
        else:
            print(f"  Result: ⚠️  Cannot evaluate (answer error)")

        print(f"\n✨ Structured Reformulated Query:")
        print(f"  {structured['reformulated_query']}")        # Show progression
        if structured['progression'] is not None:
            prog = structured['progression']
            if prog > 0:
                print(f"\n📈 Structured Rank Progression: {baseline_rank} → {structured['new_rank']} (↑{prog} positions)")
            elif prog < 0:
                print(f"\n📉 Structured Rank Degradation: {baseline_rank} → {structured['new_rank']} (↓{abs(prog)} positions)")
            else:
                print(f"\n🔄 Structured Rank Unchanged: {baseline_rank} → {structured['new_rank']}")
        elif structured['new_rank'] is not None:
            print(f"\n🎯 Structured New Rank: {structured['new_rank']}")
        
        print(f"\n🔍 Structured Stage 2 — Top 3:")
        for item in structured["stage2_top3"]:
            print(f"  #{item['rank']:>1}  📄 {item['filename']}  📊 {item['score']:.4f}")
            print(f"      🤖 BLIP: {item.get('blip_caption', 'N/A')}")
        
        # ==================== COMPARISON ====================
        print("\n" + "=" * 80)
        print("📊 PROGRESSION COMPARISON")
        print("=" * 80)
        
        if baseline_rank and unstructured['new_rank'] and structured['new_rank']:
            print(f"\nBaseline Rank: {baseline_rank}")
            print(f"Unstructured Method: {baseline_rank} → {unstructured['new_rank']} ({'+' if unstructured['progression'] > 0 else ''}{unstructured['progression']})")
            print(f"Structured Method:   {baseline_rank} → {structured['new_rank']} ({'+' if structured['progression'] > 0 else ''}{structured['progression']})")
            
            if unstructured['progression'] > structured['progression']:
                print(f"\n🏆 Unstructured method performed better (+{unstructured['progression'] - structured['progression']} advantage)")
            elif structured['progression'] > unstructured['progression']:
                print(f"\n🏆 Structured method performed better (+{structured['progression'] - unstructured['progression']} advantage)")
            else:
                print(f"\n🤝 Both methods achieved the same progression")
        
        print("=" * 80 + "\n")

# --------------------------------- CLI -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Combined Structured + Unstructured Image Retrieval")
    parser.add_argument('--vectordb', required=True, help='VectorDB name, e.g., COCO_VectorDB')
    parser.add_argument('--vectordb-dir', default='VectorDBs', help='Directory with .index and *_metadata.json')
    parser.add_argument('--prompt', type=str, help='Text prompt for image retrieval')
    parser.add_argument('--filename', type=str, help='Use true caption of this filename as query')
    parser.add_argument('--output', type=str, help='Optional: Save results to JSON file')
    # LLM parameters
    parser.add_argument('--llm', default='gemma3:4b', help='LLM model for reformulation')
    parser.add_argument('--llm-temp', type=float, default=0.1, help='LLM temperature')
    parser.add_argument('--llm-top-p', type=float, default=0.9, help='LLM top_p')
    parser.add_argument('--llm-timeout', type=int, default=200, help='LLM timeout in seconds')
    
    args = parser.parse_args()
    
    # Create retriever
    retriever = CombinedImageRetriever(
        vectordb_name=args.vectordb,
        vectordb_dir=args.vectordb_dir,
        llm_model=args.llm,
        llm_temp=args.llm_temp,
        llm_top_p=args.llm_top_p,
        llm_timeout=args.llm_timeout
    )
    
    # Determine query text and get true caption/baseline rank if filename provided
    true_caption = None
    
    if args.filename:
        true_caption, _ = retriever.get_true_caption_and_rank(args.filename)
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
    
    # Run combined retrieval
    logger.info(f"🔍 Combined retrieval for: '{query_text}'")
    result = retriever.combined_two_stage_retrieve(query_text, target_filename=args.filename)
    retriever.display_combined_results(result, true_caption, target_filename=args.filename)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "vectordb": args.vectordb,
            "filename": args.filename,
            "true_caption": true_caption,
            "llm_model": args.llm,
            **result
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()
