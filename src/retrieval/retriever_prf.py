#!/usr/bin/env python3
"""
Simple Image Retrieval with BLIP Descriptions
Given an image from a dataset, retrieve the top 10 most similar images and generate BLIP descriptions.
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRetriever:
    """
    Simple image retrieval using CLIP embeddings and FAISS index.
    Retrieves top k similar images and generates BLIP descriptions for them.
    """

    def __init__(
        self,
        vectordb_name: str,
        vectordb_dir: str = "VectorDBs",
        clip_model: str = "openai/clip-vit-large-patch14",
        llm_model: str = "gemma3:4b",
        llm_url: str = "http://localhost:11434/api/generate",
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.0
    ):
        """
        Initialize the retriever.

        Args:
            vectordb_name: Name of the VectorDB (e.g., 'COCO_VectorDB')
            vectordb_dir: Directory containing the .index and _metadata.json files
            clip_model: CLIP model to use for encoding
            llm_model: Ollama LLM model for relevance evaluation
            alpha: Rocchio alpha parameter (weight for original query, default: 1.0)
            beta: Rocchio beta parameter (weight for relevant documents, default: 0.75)
            gamma: Rocchio gamma parameter (weight for non-relevant documents, default: 0.0)
        """
        self.vectordb_name = vectordb_name
        self.vectordb_dir = Path(vectordb_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model = llm_model
        self.llm_url = llm_url

        # Rocchio parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        logger.info("=" * 60)
        logger.info("🔄 Initializing Image Retriever")
        logger.info(f"   📂 VectorDB: {vectordb_name}")
        logger.info(f"   💻 Device: {self.device}")
        logger.info(f"   🤖 LLM Model: {llm_model}")
        logger.info(f"   🌐 LLM URL: {llm_url}")
        logger.info(f"   ⚙️  Rocchio params: α={alpha}, β={beta}, γ={gamma}")
        logger.info("=" * 60)        # Load CLIP model
        logger.info(f"🔄 Loading CLIP model: {clip_model}")
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        logger.info("✅ CLIP model loaded")
        
        # Load BLIP model for image captioning
        logger.info("🔄 Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        logger.info("✅ BLIP model loaded")
        
        # Load FAISS index and metadata
        self._load_vectordb()
        
        # Set up images directory
        dataset_name = self.vectordb_name.replace("_VectorDB", "")
        self.images_dir = Path("data") / dataset_name / "images"
        logger.info(f"📂 Images directory: {self.images_dir}")
        
        # Load dataset metadata for captions and baseline ranks
        dataset_metadata_path = Path("data") / dataset_name / f"{dataset_name.lower()}_metadata.json"
        if dataset_metadata_path.exists():
            with open(dataset_metadata_path, 'r', encoding='utf-8') as f:
                self.dataset_metadata = json.load(f)
            logger.info(f"📋 Dataset metadata loaded from {dataset_metadata_path}")
        else:
            logger.warning(f"⚠️ Dataset metadata not found: {dataset_metadata_path}")
            self.dataset_metadata = None
        
        logger.info("✅ Image Retriever initialized")
        logger.info("=" * 60)
    
    def _load_vectordb(self):
        """Load FAISS index and metadata."""
        logger.info(f"🔄 Loading VectorDB: {self.vectordb_name}")
        
        # Load FAISS index
        index_path = self.vectordb_dir / f"{self.vectordb_name}.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(f"📊 FAISS index loaded: {self.faiss_index.ntotal} vectors")
        
        # Load metadata
        metadata_path = self.vectordb_dir / f"{self.vectordb_name}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_json = json.load(f)
        
        # Handle both formats: direct list or {"metadata": [...]}
        if isinstance(metadata_json, list):
            self.metadata = metadata_json
        elif isinstance(metadata_json, dict) and 'metadata' in metadata_json:
            self.metadata = metadata_json['metadata']
        else:
            raise ValueError("Invalid metadata format")
        
        logger.info(f"📋 Metadata loaded: {len(self.metadata)} entries")
        
        # Verify consistency
        if self.faiss_index.ntotal != len(self.metadata):
            logger.warning(
                f"⚠️ Mismatch: FAISS has {self.faiss_index.ntotal} vectors, "
                f"metadata has {len(self.metadata)} entries"
            )
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using CLIP.
        
        Args:
            text: Text to encode
            
        Returns:
            Normalized text embedding
        """
        try:
            with torch.no_grad():
                inputs = self.clip_processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                text_embedding = self.clip_model.get_text_features(**inputs)
                text_embedding = text_embedding.cpu().numpy().astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(text_embedding)
            return text_embedding
        except Exception as e:
            logger.error(f"❌ Failed to encode text: {e}")
            return None
    
    def encode_image(self, image_path: Path) -> np.ndarray:
        """
        Encode an image using CLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image embedding
        """
        try:
            image = Image.open(image_path).convert('RGB')
            with torch.no_grad():
                inputs = self.clip_processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                image_embedding = self.clip_model.get_image_features(**inputs)
                image_embedding = image_embedding.cpu().numpy().astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(image_embedding)
            return image_embedding
        except Exception as e:
            logger.error(f"❌ Failed to encode image {image_path}: {e}")
            return None
    
    def generate_blip_description(self, image_path: Path) -> str:
        """
        Generate a description for an image using BLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Generated description
        """
        try:
            image = Image.open(image_path).convert('RGB')
            with torch.no_grad():
                inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
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
                description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return description.strip()
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate BLIP description for {image_path}: {e}")
            return "Failed to generate description"
    
    def evaluate_relevance_with_llm(self, query_description: str, results: List[Dict]) -> Dict:
        """
        Use LLM to evaluate which retrieved images are relevant to the query.
        Includes retry mechanism with up to 3 attempts.
        
        Args:
            query_description: BLIP description of the query image
            results: List of retrieval results with BLIP descriptions
            
        Returns:
            Dictionary containing LLM evaluation and relevant indices
        """
        logger.info("🤖 Evaluating relevance with LLM...")
        
        # Build prompt for LLM
        prompt = f"""You are an expert image relevance evaluator. Given a query image description and 10 retrieved image descriptions, determine which retrieved images are relevant to the query.

QUERY IMAGE DESCRIPTION:
"{query_description}"

RETRIEVED IMAGES:
"""
        
        for result in results:
            prompt += f"\nIndex {result['rank']}: {result['blip_description']}\n"
        
        prompt += """
Based on semantic similarity and relevance, which indices (1-10) represent images that are relevant to the query image?

Respond ONLY with a JSON object in this exact format:
{
  "relevant_indices": [list of relevant indices as integers],
  "reasoning": "brief explanation of your decision"
}

Be selective - only mark images as relevant if they share significant semantic content with the query."""

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Call Ollama LLM using subprocess and curl
                logger.info(f"🤖 Calling {self.llm_model} via Ollama API... (Attempt {attempt}/{max_retries})")
                
                request_data = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                }

                print(f"LLM URL: {self.llm_url}")
                
                cmd = [
                    "curl", "-s", "-X", "POST", 
                    f"{self.llm_url}",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(request_data)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                
                if result.returncode != 0:
                    raise Exception(f"Curl failed: {result.stderr}")
                
                response = json.loads(result.stdout)
                llm_response = response.get('response', '').strip()
                
                if not llm_response:
                    raise Exception("Empty response from LLM")
                
                logger.info(f"📝 LLM Response: {llm_response}")
                
                # Parse JSON response
                import re
                json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                    
                    # Validate the response structure
                    if 'relevant_indices' not in evaluation:
                        raise ValueError("Missing 'relevant_indices' in LLM response")
                    
                    if not isinstance(evaluation['relevant_indices'], list):
                        raise ValueError("'relevant_indices' must be a list")
                    
                    # Validate indices are within valid range (1-10)
                    for idx in evaluation['relevant_indices']:
                        if not isinstance(idx, int) or idx < 1 or idx > 10:
                            raise ValueError(f"Invalid index {idx} - must be integer between 1-10")
                    
                    logger.info(f"✅ LLM Evaluation: {evaluation['relevant_indices']}")
                    logger.info(f"💡 Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")
                    return evaluation
                else:
                    raise ValueError("Could not parse LLM response as JSON")
                    
            except Exception as e:
                logger.warning(f"⚠️ LLM evaluation attempt {attempt} failed: {e}")
                
                if attempt == max_retries:
                    logger.error(f"❌ All {max_retries} LLM evaluation attempts failed")
                    return {
                        "relevant_indices": [],
                        "reasoning": f"Error after {max_retries} attempts: {str(e)}",
                        "error": True,
                        "attempts": max_retries
                    }
                else:
                    logger.info(f"🔄 Retrying LLM evaluation in 2 seconds...")
                    import time
                    time.sleep(2)
        
        # This should never be reached due to the logic above, but just in case
        return {
            "relevant_indices": [],
            "reasoning": "Unexpected error in retry logic",
            "error": True
        }
    
    def get_caption_for_image(self, target_filename: str) -> Optional[tuple]:
        """
        Get the true caption and baseline_rank for a target image from dataset metadata.
        
        Args:
            target_filename: Filename or path of the target image
            
        Returns:
            Tuple of (caption, baseline_rank) or None if not found
        """
        # Extract just the filename if a path was provided
        if '/' in target_filename or '\\' in target_filename:
            target_filename = os.path.basename(target_filename)
        
        logger.info(f"🔍 Looking for caption for: {target_filename}")
        
        if not self.dataset_metadata:
            logger.error("❌ Dataset metadata not loaded")
            return None
        
        # Handle different metadata formats
        # Format 1: List of dictionaries with 'images' key
        if isinstance(self.dataset_metadata, dict) and 'images' in self.dataset_metadata:
            metadata_list = self.dataset_metadata['images']
        # Format 2: Direct list
        elif isinstance(self.dataset_metadata, list):
            metadata_list = self.dataset_metadata
        else:
            logger.error(f"❌ Unexpected metadata format: {type(self.dataset_metadata)}")
            return None
        
        # Search in dataset metadata
        for entry in metadata_list:
            # Skip if entry is not a dict
            if not isinstance(entry, dict):
                continue
                
            filename = entry.get('filename', '')
            # Also try 'image_path' field
            if not filename:
                image_path = entry.get('image_path', '')
                if image_path:
                    filename = os.path.basename(image_path)
            
            # Handle both with and without extension
            if filename == target_filename or os.path.splitext(filename)[0] == os.path.splitext(target_filename)[0]:
                caption = entry.get('caption', '')
                baseline_rank = entry.get('baseline_rank')
                logger.info(f"📝 Found caption: {caption}")
                if baseline_rank is not None:
                    logger.info(f"📊 Baseline rank in dataset: {baseline_rank}")
                return (caption, baseline_rank)
        
        logger.error(f"❌ Caption not found for {target_filename}")
        logger.error(f"   Available fields in first entry: {list(metadata_list[0].keys()) if metadata_list and isinstance(metadata_list[0], dict) else 'N/A'}")
        return None
    
    def apply_rocchio_prf(self, query_embedding: np.ndarray, relevant_results: List[Dict], 
                          alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.0) -> np.ndarray:
        """
        Apply Rocchio algorithm for pseudo-relevance feedback.
        
        Args:
            query_embedding: Original query embedding
            relevant_results: List of results marked as relevant by LLM
            alpha: Weight for original query (default: 1.0)
            beta: Weight for relevant documents (default: 0.75)
            gamma: Weight for non-relevant documents (default: 0.0)
            
        Returns:
            Modified query embedding
        """
        logger.info(f"🔄 Applying Rocchio PRF with α={alpha}, β={beta}, γ={gamma}")
        logger.info(f"   Using {len(relevant_results)} relevant images")
        
        if not relevant_results:
            logger.warning("⚠️ No relevant results for PRF, returning original query")
            return query_embedding
        
        # Get embeddings of relevant images
        relevant_embeddings = []
        for result in relevant_results:
            image_path = Path(result['image_path'])
            if image_path.exists():
                emb = self.encode_image(image_path)
                if emb is not None:
                    relevant_embeddings.append(emb)
        
        if not relevant_embeddings:
            logger.warning("⚠️ Could not encode relevant images, returning original query")
            return query_embedding
        
        # Calculate centroid of relevant documents
        relevant_embeddings_array = np.vstack(relevant_embeddings)
        relevant_centroid = np.mean(relevant_embeddings_array, axis=0, keepdims=True)
        
        # Apply Rocchio formula: Q' = α*Q + β*(1/|Dr|)*Σ(Dr) - γ*(1/|Dnr|)*Σ(Dnr)
        # With γ=0, we ignore non-relevant documents
        modified_query = alpha * query_embedding + beta * relevant_centroid
        
        # Normalize for cosine similarity
        faiss.normalize_L2(modified_query)
        
        logger.info("✅ Rocchio PRF applied successfully")
        return modified_query
    
    def retrieve_with_caption(self, caption: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top k images using a text caption.
        
        Args:
            caption: Text caption to search with
            k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        logger.info(f"🔍 Retrieving top {k} similar images using caption")
        logger.info(f"📝 Caption: {caption}")
        
        # Encode caption with CLIP
        query_embedding = self.encode_text(caption)
        if query_embedding is None:
            logger.error("❌ Failed to encode caption")
            return []
        
        # Search in FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        logger.info(f"🔎 Search complete")
        logger.info(f"   Top similarity: {similarities[0][0]:.4f}")
        
        # Build results
        results = []
        for rank, (similarity, faiss_idx) in enumerate(zip(similarities[0], indices[0]), 1):
            # Find metadata by embedding_id
            metadata_entry = None
            for entry in self.metadata:
                if entry.get('embedding_id') == int(faiss_idx):
                    metadata_entry = entry
                    break
            
            if metadata_entry:
                filename = metadata_entry.get('filename', f'image_{faiss_idx}.jpg')
                image_path = self.images_dir / filename
                
                # Generate BLIP description
                blip_description = "Image not found"
                if image_path.exists():
                    blip_description = self.generate_blip_description(image_path)
                
                result = {
                    'rank': rank,
                    'faiss_index': int(faiss_idx),
                    'similarity': float(similarity),
                    'filename': filename,
                    'original_caption': metadata_entry.get('caption', 'No caption available'),
                    'blip_description': blip_description,
                    'image_path': str(image_path) if image_path.exists() else None
                }
                results.append(result)
        
        return results
    
    def retrieve_with_prf(self, query_caption: str, baseline_results: List[Dict], 
                          llm_evaluation: Dict, target_filename: str, k: int = 10) -> Dict:
        """
        Perform re-retrieval using pseudo-relevance feedback and find exact rank of target image.
        
        Args:
            query_caption: Original text caption query
            baseline_results: Initial retrieval results
            llm_evaluation: LLM evaluation with relevant indices
            target_filename: Filename of the target image to track
            k: Number of top results to display (but will search exhaustively for target)
            
        Returns:
            Dictionary with PRF results, target tracking, and comparison
        """
        logger.info("=" * 60)
        logger.info("🔄 PSEUDO-RELEVANCE FEEDBACK RE-RETRIEVAL")
        logger.info("=" * 60)
        
        # Get relevant results based on LLM evaluation
        relevant_indices = llm_evaluation.get('relevant_indices', [])
        
        if not relevant_indices:
            logger.warning("⚠️ No relevant images identified by LLM, skipping PRF")
            return None
        
        # Filter relevant results
        relevant_results = [r for r in baseline_results if r['rank'] in relevant_indices]
        logger.info(f"📋 Using {len(relevant_results)} relevant images for PRF")
        
        # Encode original query caption
        query_embedding = self.encode_text(query_caption)
        if query_embedding is None:
            logger.error("❌ Failed to encode query caption")
            return None

        # Apply Rocchio PRF
        modified_query = self.apply_rocchio_prf(query_embedding, relevant_results,
                                                alpha=self.alpha, beta=self.beta, gamma=self.gamma)

        # Re-retrieve ALL images to find exact rank of target
        logger.info(f"🔍 Re-retrieving ALL images to find exact rank of target...")
        total_images = self.faiss_index.ntotal
        similarities, indices = self.faiss_index.search(modified_query, total_images)
        
        logger.info(f"🔎 PRF search complete - searched {total_images} images")
        logger.info(f"   Top similarity: {similarities[0][0]:.4f}")
        
        # Find target image's exact rank
        target_prf_rank = None
        target_prf_similarity = None
        target_faiss_idx = None
        
        for rank, (similarity, faiss_idx) in enumerate(zip(similarities[0], indices[0]), 1):
            # Find metadata by embedding_id
            metadata_entry = None
            for entry in self.metadata:
                if entry.get('embedding_id') == int(faiss_idx):
                    metadata_entry = entry
                    break
            
            if metadata_entry:
                filename = metadata_entry.get('filename', f'image_{faiss_idx}.jpg')
                
                # Check if this is the target image
                if filename == target_filename:
                    target_prf_rank = rank
                    target_prf_similarity = float(similarity)
                    target_faiss_idx = int(faiss_idx)
                    logger.info(f"🎯 Target image found at EXACT PRF rank #{target_prf_rank} (similarity: {target_prf_similarity:.4f})")
                    break
        
        if target_prf_rank is None:
            logger.warning(f"⚠️ Target image '{target_filename}' not found in PRF results")
        
        # Build top-K PRF results for display
        prf_top_k = []
        for rank in range(1, min(k + 1, total_images + 1)):
            similarity = similarities[0][rank - 1]
            faiss_idx = indices[0][rank - 1]
            
            # Find metadata by embedding_id
            metadata_entry = None
            for entry in self.metadata:
                if entry.get('embedding_id') == int(faiss_idx):
                    metadata_entry = entry
                    break
            
            if metadata_entry:
                filename = metadata_entry.get('filename', f'image_{faiss_idx}.jpg')
                image_path = self.images_dir / filename
                
                # Generate BLIP description for display
                blip_description = "Image not found"
                if image_path.exists():
                    blip_description = self.generate_blip_description(image_path)
                
                result = {
                    'prf_rank': rank,
                    'faiss_index': int(faiss_idx),
                    'similarity': float(similarity),
                    'filename': filename,
                    'original_caption': metadata_entry.get('caption', 'No caption available'),
                    'blip_description': blip_description,
                    'is_target': filename == target_filename
                }
                prf_top_k.append(result)
        
        logger.info(f"✅ Retrieved top-{k} results for display")
        
        return {
            'prf_top_k': prf_top_k,
            'target_prf_rank': target_prf_rank,
            'target_prf_similarity': target_prf_similarity,
            'target_faiss_idx': target_faiss_idx,
            'num_relevant_used': len(relevant_results),
            'relevant_indices': relevant_indices,
            'total_searched': total_images
        }
    
    def retrieve_top_k(self, query_image_path: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top k most similar images using CLIP image encoding.
        
        Args:
            query_image_path: Path to the query image
            k: Number of results to return
            
        Returns:
            List of dictionaries containing image information, similarity scores, and BLIP descriptions
        """
        logger.info(f"🔍 Retrieving top {k} similar images")
        logger.info(f"�️ Query Image: {query_image_path}")
        
        # Encode query image with CLIP
        query_embedding = self.encode_image(Path(query_image_path))
        if query_embedding is None:
            logger.error("❌ Failed to encode query image")
            return []
        
        # Search in FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        logger.info(f"🔎 Search complete")
        logger.info(f"   Top similarity: {similarities[0][0]:.4f}")
        logger.info(f"   FAISS indices: {indices[0][:5]}...")
        
        # Build results
        results = []
        logger.info("🖼️ Generating BLIP descriptions for retrieved images...")
        
        for rank, (similarity, faiss_idx) in enumerate(zip(similarities[0], indices[0]), 1):
            # Find metadata by embedding_id
            metadata_entry = None
            for entry in self.metadata:
                if entry.get('embedding_id') == int(faiss_idx):
                    metadata_entry = entry
                    break
            
            if metadata_entry:
                filename = metadata_entry.get('filename', f'image_{faiss_idx}.jpg')
                image_path = self.images_dir / filename
                
                # Generate BLIP description
                blip_description = "Image not found"
                if image_path.exists():
                    blip_description = self.generate_blip_description(image_path)
                    logger.info(f"   #{rank}: {filename} - {blip_description}...")
                else:
                    logger.warning(f"   ⚠️ Image file not found: {image_path}")
                
                result = {
                    'rank': rank,
                    'faiss_index': int(faiss_idx),
                    'similarity': float(similarity),
                    'filename': filename,
                    'original_caption': metadata_entry.get('caption', 'No caption available'),
                    'blip_description': blip_description,
                    'image_path': str(image_path) if image_path.exists() else None
                }
                results.append(result)
            else:
                logger.warning(f"⚠️ No metadata found for embedding_id {faiss_idx}")
        
        logger.info(f"✅ Retrieved {len(results)} results with BLIP descriptions")
        return results
    
    def retrieve_from_image(self, target_filename: str, k: int = 10) -> Optional[Dict]:
        """
        Given a target image filename, use its caption to retrieve similar images,
        then apply PRF and track the target image's rank progression.
        
        Args:
            target_filename: Filename of the target image (e.g., "000000000077.jpg")
            k: Number of results to return
            
        Returns:
            Dictionary containing query info, baseline results, PRF results, and target rank progression
        """
        logger.info("=" * 80)
        logger.info("🎯 IMAGE RETRIEVAL WITH PSEUDO-RELEVANCE FEEDBACK")
        logger.info("=" * 80)
        logger.info(f"📷 Target Image: {target_filename}")
        
        # Extract filename if path was provided
        if '/' in target_filename or '\\' in target_filename:
            target_filename = os.path.basename(target_filename)
            logger.info(f"📝 Extracted filename: {target_filename}")
        
        # Get the true caption for this image
        caption_data = self.get_caption_for_image(target_filename)
        if not caption_data:
            logger.error(f"❌ Cannot proceed without caption")
            return None
        
        caption, dataset_baseline_rank = caption_data
        logger.info(f"📝 Using true caption as query: {caption}...")
        
        # Baseline retrieval using the caption
        logger.info("\n" + "=" * 60)
        logger.info("🔍 BASELINE RETRIEVAL (using true caption)")
        logger.info("=" * 60)
        baseline_results = self.retrieve_with_caption(caption, k)
        
        if not baseline_results:
            logger.error("❌ Baseline retrieval failed")
            return None
        
        # Check if target image is in top-k results
        target_in_topk = False
        target_topk_rank = None
        target_baseline_similarity = None
        
        for result in baseline_results:
            if result['filename'] == target_filename:
                target_in_topk = True
                target_topk_rank = result['rank']
                target_baseline_similarity = result['similarity']
                logger.info(f"\n🎯 Target image found in top-{k} at rank #{target_topk_rank} (similarity: {target_baseline_similarity:.4f})")
                break
        
        if not target_in_topk:
            logger.info(f"ℹ️ Target image '{target_filename}' not in baseline top-{k}")
            logger.info(f"📊 Using baseline rank from dataset metadata: #{dataset_baseline_rank}")
        
        # Generate BLIP descriptions for LLM evaluation
        logger.info(f"\n🖼️ Generating BLIP descriptions for LLM evaluation...")
        for result in baseline_results:
            # BLIP description already generated in retrieve_with_caption
            pass
        
        # Evaluate relevance with LLM
        llm_evaluation = self.evaluate_relevance_with_llm(caption, baseline_results)
        
        # Apply PRF if LLM found relevant images
        prf_data = None
        target_prf_rank = None
        target_prf_similarity = None
        rank_improvement = None
        
        if llm_evaluation and llm_evaluation.get('relevant_indices'):
            prf_data = self.retrieve_with_prf(caption, baseline_results, llm_evaluation, target_filename, k)
            
            if prf_data:
                # Get exact PRF rank from exhaustive search
                target_prf_rank = prf_data.get('target_prf_rank')
                target_prf_similarity = prf_data.get('target_prf_similarity')
                
                if target_prf_rank is not None and dataset_baseline_rank is not None:
                    rank_improvement = dataset_baseline_rank - target_prf_rank
                    logger.info(f"\n🎯 Target image exact PRF rank: #{target_prf_rank} (similarity: {target_prf_similarity:.4f})")
                    
                    if rank_improvement > 0:
                        logger.info(f"📈 IMPROVEMENT: Rank improved by {rank_improvement} positions ({dataset_baseline_rank} → {target_prf_rank})")
                    elif rank_improvement < 0:
                        logger.info(f"📉 DEGRADATION: Rank dropped by {abs(rank_improvement)} positions ({dataset_baseline_rank} → {target_prf_rank})")
                    else:
                        logger.info(f"➡️  NO CHANGE: Rank remained at #{target_prf_rank}")
                elif target_prf_rank is None:
                    logger.warning(f"⚠️ Target image '{target_filename}' not found in PRF results")
        
        return {
            'target_filename': target_filename,
            'query_caption': caption,
            'num_results': k,
            'baseline_results': baseline_results,
            'llm_evaluation': llm_evaluation,
            'prf_data': prf_data,
            'target_tracking': {
                'baseline_rank': dataset_baseline_rank,  # Use baseline rank from dataset metadata
                'baseline_similarity': target_baseline_similarity,
                'prf_rank': target_prf_rank,
                'prf_similarity': target_prf_similarity,
                'improvement': rank_improvement,
                'in_baseline_topk': target_in_topk,
                'baseline_topk_rank': target_topk_rank
            }
        }
    
    def display_results(self, retrieval_result: Dict):
        """Display retrieval results with target image rank tracking."""
        if not retrieval_result:
            return
        
        print("\n" + "=" * 80)
        print("🎯 IMAGE RETRIEVAL WITH PSEUDO-RELEVANCE FEEDBACK")
        print("=" * 80)
        print(f"📷 Target Image: {retrieval_result['target_filename']}")
        print(f"📝 Query: {retrieval_result['query_caption']}...")
        print("=" * 80)
        
        target_tracking = retrieval_result.get('target_tracking', {})
        baseline_results = retrieval_result.get('baseline_results', [])
        llm_eval = retrieval_result.get('llm_evaluation', {})
        prf_data = retrieval_result.get('prf_data')
        
        # 1. BASELINE TOP 10 with BLIP descriptions
        print(f"\n📊 BASELINE RETRIEVAL - TOP 10:")
        print("-" * 80)
        
        for result in baseline_results:
            is_target = "🎯 " if result['filename'] == retrieval_result['target_filename'] else "   "
            
            print(f"{is_target}#{result['rank']}: {result['filename']}")
            print(f"      📝 BLIP: {result.get('blip_description', 'N/A')}")
            print("-" * 80)
        
        # 2. LLM DECISION
        if llm_eval:
            print(f"\n🤖 LLM RELEVANCE DECISION:")
            print("-" * 80)
            relevant_indices = llm_eval.get('relevant_indices', [])
            reasoning = llm_eval.get('reasoning', 'No reasoning provided')
            
            print(f"✅ Relevant images: {relevant_indices}")
            print(f"💡 Reasoning: {reasoning}")
            print("-" * 80)
        
        # 3. PRF TOP 10 with BLIP descriptions
        if prf_data:
            prf_top_k = prf_data.get('prf_top_k', [])
            
            if prf_top_k:
                print(f"\n🔄 AFTER PRF - TOP 10:")
                print("-" * 80)
                
                for result in prf_top_k:
                    is_target = "🎯 " if result.get('is_target', False) else "   "
                    
                    print(f"{is_target}#{result['prf_rank']}: {result['filename']}")
                    print(f"      📝 BLIP: {result.get('blip_description', 'N/A')}")
                    print("-" * 80)
        
        # 4. TARGET IMAGE EXACT RANK PROGRESSION
        print(f"\n🎯 TARGET IMAGE RANK PROGRESSION:")
        print("=" * 80)
        
        baseline_rank = target_tracking.get('baseline_rank')
        prf_rank = target_tracking.get('prf_rank')
        improvement = target_tracking.get('improvement')
        
        print(f"Baseline Rank: #{baseline_rank if baseline_rank is not None else 'Not found'}")
        
        if prf_rank is not None:
            print(f"PRF Rank (exact): #{prf_rank}")
            
            if improvement is not None and baseline_rank is not None:
                if improvement > 0:
                    print(f"✅ IMPROVEMENT: +{improvement} positions (#{baseline_rank} → #{prf_rank})")
                elif improvement < 0:
                    print(f"❌ DEGRADATION: {improvement} positions (#{baseline_rank} → #{prf_rank})")
                else:
                    print(f"➡️  NO CHANGE")
        else:
            print(f"PRF Rank: Not found")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Image Retrieval with PRF: "
                    "Given a target image, use its caption to retrieve similar images and apply PRF"
    )
    parser.add_argument(
        '--vectordb',
        required=True,
        help='VectorDB name (e.g., COCO_VectorDB, Flickr_VectorDB, VizWiz_VectorDB)'
    )
    parser.add_argument(
        '--filename',
        required=True,
        help='Target image filename (e.g., 000000000077.jpg)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to retrieve (default: 10)'
    )
    parser.add_argument(
        '--vectordb-dir',
        default='VectorDBs',
        help='Directory containing VectorDB files (default: VectorDBs)'
    )
    parser.add_argument(
        '--llm-model',
        default='gemma3:4b',
        help='Ollama LLM model for relevance evaluation (default: gemma3:4b)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize retriever
        retriever = ImageRetriever(
            vectordb_name=args.vectordb,
            vectordb_dir=args.vectordb_dir,
            llm_model=args.llm_model
        )
        
        # Perform retrieval with PRF
        result = retriever.retrieve_from_image(args.filename, k=args.k)
        
        # Display results
        if result:
            retriever.display_results(result)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
