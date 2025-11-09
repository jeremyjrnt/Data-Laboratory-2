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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_retrieval")

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
        llm_model: str = "gemma3:4b",
        llm_temp: float = 0.1,
        llm_top_p: float = 0.9,
        llm_timeout: int = 240
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
    
    def _call_ollama_llm(self, prompt: str) -> str:
        """Call Ollama LLM to reformulate query."""
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
            logger.error(f"❌ LLM call failed: {e}")
            return ""
    
    def _build_reformulation_prompt(self, original_query: str, top3_results: List[Dict]) -> str:
        """Build prompt for LLM to reformulate the query based on BLIP descriptions."""
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
    
    def reformulate_query_with_llm(self, original_query: str, top3_results: List[Dict]) -> str:
        """Use LLM to reformulate query based on BLIP descriptions of top 3 results."""
        prompt = self._build_reformulation_prompt(original_query, top3_results)
        reformulated = self._call_ollama_llm(prompt)
        
        if not reformulated:
            logger.warning("⚠️ LLM reformulation failed, using original query")
            return original_query
        
        # Extract just the reformulated query (remove any extra text)
        lines = [line.strip() for line in reformulated.strip().split('\n') if line.strip()]
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
        Two-stage retrieval: CLIP -> BLIP -> LLM reformulation -> CLIP again.
        Always returns top 3 results for both stages with BLIP captions.
        
        Args:
            query_text: Original textual query
            target_filename: Filename of the target image to track its rank
            
        Returns:
            Dictionary containing:
                - original_query: Original query text
                - stage1_top3: Top 3 results from first retrieval with BLIP captions
                - reformulated_query: LLM-reformulated query
                - stage2_top3: Top 3 final results from reformulated query with BLIP captions
                - target_rank_after_reformulation: New rank of target image (if provided)
        """
        # Stage 1: Get top 3 with BLIP descriptions
        logger.info("🔍 Stage 1: Initial retrieval and BLIP captioning...")
        stage1_top3 = self.retrieve_top3(query_text)
        
        # LLM reformulation
        logger.info("🤖 LLM reformulation based on BLIP descriptions...")
        reformulated_query = self.reformulate_query_with_llm(query_text, stage1_top3)
        logger.info(f"✨ Reformulated query: '{reformulated_query}'")
        
        # Stage 2: Re-retrieval with reformulated query (always top 3 with BLIP)
        logger.info(f"🔍 Stage 2: Re-retrieval with reformulated query...")
        stage2_top3 = self.retrieve_top3(reformulated_query)
        
        # Find new rank of target image if provided
        target_rank_after = None
        if target_filename:
            logger.info(f"🎯 Finding new rank of target image: {target_filename}")
            target_rank_after = self._find_target_image_rank(reformulated_query, target_filename)
            
            # If not found, run debug
            if target_rank_after is None:
                self._debug_target_image(target_filename)
        
        return {
            "original_query": query_text,
            "stage1_top3": stage1_top3,
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
        """Display two-stage retrieval results."""
        print("\n" + "=" * 80)
        print("🔁 TWO-STAGE RETRIEVAL WITH LLM REFORMULATION")
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
        
        print("\n🔍 Stage 1 — Top 3 from initial retrieval:")
        for item in result["stage1_top3"]:
            print(f"  #{item['rank']:>1}  📄 {item['filename']}  📊 {item['score']:.4f}")
            print(f"      🤖 BLIP: {item.get('blip_caption', 'N/A')}")
        
        print(f"\n✨ LLM Reformulated Query:")
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
    parser.add_argument('--llm', default='gemma3:27b', help='LLM model for reformulation (default: gemma3:27b)')
    parser.add_argument('--llm-temp', type=float, default=0.1, help='LLM temperature (default: 0.1)')
    parser.add_argument('--llm-top-p', type=float, default=0.9, help='LLM top_p (default: 0.9)')
    parser.add_argument('--llm-timeout', type=int, default=120, help='LLM timeout in seconds (default: 120)')
    
    args = parser.parse_args()
    
    # Create retriever
    retriever = ImageRetriever(
        vectordb_name=args.vectordb,
        vectordb_dir=args.vectordb_dir,
        llm_model=args.llm,
        llm_temp=args.llm_temp,
        llm_top_p=args.llm_top_p,
        llm_timeout=args.llm_timeout
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
