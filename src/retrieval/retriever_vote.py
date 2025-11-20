#!/usr/bin/env python3
"""
Ground Truth LLM Voting Retriever
Advanced image retrieval system that:
1. Uses CLIP for initial similarity search
2. Generates BLIP descriptions for retrieved images  
3. Uses LLM to rerank based on semantic understanding
4. Evaluates performance on ground truth caption-image pairs
"""

import os
import sys
import json
import subprocess
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

import torch
import numpy as np
import faiss
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration
)
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundTruthLLMVotingRetriever:
    def __init__(self, vectordb_name: str, vectordb_dir: str = None, llm_model: str = None):
        self.vectordb_name = vectordb_name
        self.vectordb_dir = Path(vectordb_dir) if vectordb_dir else Config.VECTORDB_DIR
        self.llm_model = llm_model or Config.OLLAMA_MODEL_LARGE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        logger.info("🔄 Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(Config.HF_MODEL_CLIP_LARGE).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(Config.HF_MODEL_CLIP_LARGE)
        logger.info("✅ CLIP model loaded")
        
        logger.info("🔄 Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained(Config.HF_MODEL_BLIP)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(Config.HF_MODEL_BLIP).to(self.device)
        logger.info("✅ BLIP model loaded")
        
        # Load VectorDB
        self._load_vectordb()
        
        logger.info("✅ Ground Truth LLM Voting Retriever initialized")
        logger.info(f"   🎯 VectorDB: {self.vectordb_name}")
        logger.info(f"   🤖 LLM: {self.llm_model}")
        logger.info(f"   💻 Device: {self.device}")
    
    def _load_vectordb(self):
        """Load FAISS index and metadata."""
        logger.info(f"🔄 Loading VectorDB: {self.vectordb_name}")
        
        # Load FAISS index (different naming convention)
        index_path = self.vectordb_dir / f"{self.vectordb_name}.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(f"📊 FAISS index loaded: {self.faiss_index.ntotal} vectors")
        
        # Load metadata (different naming convention)
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
        expected_vectors = len(self.metadata)
        if self.faiss_index.ntotal != expected_vectors:
            logger.warning(f"⚠️ Mismatch: FAISS has {self.faiss_index.ntotal} vectors, metadata has {expected_vectors} entries")
        else:
            logger.info(f"📊 Expected vectors: {expected_vectors}")
        
        # Set up images directory
        self.images_dir = Path("data") / self.vectordb_name.replace("_VectorDB", "") / "images"
        logger.info(f"📂 Image directory: {self.images_dir}")
        
        if not self.images_dir.exists():
            logger.warning(f"⚠️ Image directory not found: {self.images_dir}")
    
    def get_random_ground_truth_caption(self) -> Tuple[str, int, Dict]:
        """Get a random caption and its corresponding ground truth image ID."""
        # Select random metadata entry
        random_entry = random.choice(self.metadata)
        
        caption = random_entry['caption']
        embedding_id = random_entry['embedding_id']
        
        logger.info(f"🎲 Selected random ground truth:")
        logger.info(f"   📋 Embedding ID: {embedding_id}")
        logger.info(f"   📄 Filename: {random_entry.get('filename', 'N/A')}")
        logger.info(f"   📝 Caption: {caption[:70]}...")
        
        return caption, embedding_id, random_entry
    
    def get_caption_by_filename(self, filename: str) -> Optional[Tuple[str, int, Dict]]:
        """Get caption and ground truth for a specific filename."""
        for entry in self.metadata:
            if entry.get('filename') == filename:
                caption = entry['caption']
                embedding_id = entry['embedding_id']
                
                logger.info(f"🎯 Found ground truth for filename: {filename}")
                logger.info(f"   📋 Embedding ID: {embedding_id}")
                logger.info(f"   📝 Caption: {caption[:70]}...")
                
                return caption, embedding_id, entry
        
        logger.error(f"❌ Filename not found in metadata: {filename}")
        return None
    
    def retrieve_similar_images(self, query_text: str, k: int = 5) -> List[Dict]:
        """Retrieve k most similar images using CLIP."""
        logger.info(f"🔍 Searching for {k} similar images...")
        
        # Encode query text
        with torch.no_grad():
            inputs = self.clip_processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
            query_embedding = self.clip_model.get_text_features(**inputs)
            query_embedding = query_embedding.cpu().numpy().astype('float32')
        
        # Search in FAISS
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        # Debug information
        logger.info(f"🔎 Debug - Search results:")
        logger.info(f"   Similarities: {similarities[0]}")
        logger.info(f"   FAISS Indices: {indices[0]}")
        logger.info(f"   Metadata length: {len(self.metadata)}")
        
        results = []
        for i, (similarity, faiss_idx) in enumerate(zip(similarities[0], indices[0])):
            # Find metadata by embedding_id (which should match FAISS index)
            metadata_entry = None
            for entry in self.metadata:
                if entry.get('embedding_id') == int(faiss_idx):
                    metadata_entry = entry
                    break
            
            logger.info(f"   Processing FAISS idx {faiss_idx} (similarity: {similarity:.4f})")
            
            if metadata_entry:
                logger.info(f"     ✅ Found metadata for embedding_id {faiss_idx}")
                
                result = {
                    'temp_index': i + 1,  # Temporary 1-based index for LLM prompts
                    'clip_rank': i + 1,   # CLIP ranking (1-based)
                    'faiss_index': int(faiss_idx),  # Original FAISS index
                    'similarity': float(similarity),
                    'filename': metadata_entry.get('filename', f'image_{faiss_idx}.jpg'),
                    'original_caption': metadata_entry.get('caption', 'No caption available')
                }
                results.append(result)
                logger.info(f"     ✅ Added: temp_idx={result['temp_index']}, faiss_idx={result['faiss_index']}")
            else:
                logger.warning(f"     ❌ No metadata found for embedding_id {faiss_idx}")
        
        logger.info(f"✅ Retrieved {len(results)} similar images")
        return results
    
    def generate_blip_caption(self, image_path: Path) -> str:
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
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate BLIP caption for {image_path}: {e}")
            return "Failed to generate description"
    
    def call_ollama_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry mechanism for better reliability."""
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 LLM Call Attempt {attempt + 1}/{max_retries}")
                
                request_data = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1 if attempt == 0 else 0.2 + (attempt * 0.1),  # Increase temperature on retries
                        "top_p": 0.9
                    }
                }
                
                cmd = [
                    "curl", "-s", "-X", "POST", 
                    Config.get_ollama_url('a6000'),
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
                    
                logger.info(f"✅ LLM responded successfully (attempt {attempt + 1})")
                return llm_response
                
            except Exception as e:
                logger.warning(f"⚠️ LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ All LLM call attempts failed")
                    return "Based on similarity: [1, 2, 3, 4, 5]"
                else:
                    logger.info(f"🔄 Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
        
        return "Based on similarity: [1, 2, 3, 4, 5]"

    def call_ollama_llm(self, prompt: str) -> str:
        try:
            request_data = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            cmd = [
                "curl", "-s", "-X", "POST", 
                Config.get_ollama_url('a6000'),
                "-H", "Content-Type: application/json",
                "-d", json.dumps(request_data)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise Exception(f"Curl failed: {result.stderr}")
            
            response = json.loads(result.stdout)
            return response.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            logger.info("🔄 Using fallback ranking based on similarity")
            return "Based on similarity: [1, 2, 3, 4, 5]"
    
    def parse_llm_ranking(self, llm_response: str, num_images: int) -> List[int]:
        """Parse LLM ranking response with multiple strategies and detailed logging."""
        logger.info(f"� PARSING LLM RANKING - Detailed Analysis:")
        logger.info(f"�📝 Full LLM Response ({len(llm_response)} chars):")
        logger.info(f"'{llm_response}'")
        logger.info(f"🔢 Expected: {num_images} numbers from 1 to {num_images}")
        logger.info("-" * 50)
        
        # Strategy 1: Enhanced array patterns
        array_patterns = [
            r'\[(\d+(?:,\s*\d+)*)\]',                      # [1, 2, 3]
            r'ranking:\s*\[(\d+(?:,\s*\d+)*)\]',           # Ranking: [1, 2, 3]  
            r'order:\s*\[(\d+(?:,\s*\d+)*)\]',             # Order: [1, 2, 3]
            r'result:\s*\[(\d+(?:,\s*\d+)*)\]',            # Result: [1, 2, 3]
            r'answer:\s*\[(\d+(?:,\s*\d+)*)\]',            # Answer: [1, 2, 3]
            r'\[(\d+(?:\s*,\s*\d+)*)\]',                   # [1,2,3] or [1 , 2 , 3]
            r'^\s*(\d+(?:,\s*\d+)*)\s*$',                  # Just numbers: 1, 2, 3
        ]
        
        logger.info("🔍 Testing array patterns...")
        for i, pattern in enumerate(array_patterns, 1):
            logger.info(f"   Pattern {i}: {pattern}")
            array_match = re.search(pattern, llm_response, re.IGNORECASE | re.MULTILINE)
            if array_match:
                try:
                    numbers_str = array_match.group(1)
                    logger.info(f"   ✅ Match found: '{numbers_str}'")
                    numbers = [int(x.strip()) for x in numbers_str.split(',')]
                    logger.info(f"   📊 Parsed numbers: {numbers}")
                    
                    if len(numbers) == num_images and set(numbers) == set(range(1, num_images + 1)):
                        logger.info(f"✅ Valid ranking parsed with pattern {i}: {numbers}")
                        return numbers
                    else:
                        # Detailed validation logging
                        unique_numbers = set(numbers)
                        expected_set = set(range(1, num_images + 1))
                        has_duplicates = len(numbers) != len(unique_numbers)
                        
                        logger.warning(f"   ❌ Invalid ranking detected:")
                        logger.warning(f"      - Numbers: {numbers}")
                        logger.warning(f"      - Length: {len(numbers)}, expected: {num_images}")
                        logger.warning(f"      - Unique count: {len(unique_numbers)}")
                        logger.warning(f"      - Has duplicates: {has_duplicates}")
                        logger.warning(f"      - Expected set: {expected_set}")
                        logger.warning(f"      - Actual set: {unique_numbers}")
                        logger.warning(f"      - Missing numbers: {expected_set - unique_numbers}")
                        logger.warning(f"      - Extra/invalid numbers: {unique_numbers - expected_set}")
                        
                        # Return the invalid ranking so it can be validated by quality check
                        logger.info(f"🔄 Returning invalid ranking for quality validation: {numbers}")
                        return numbers
                except ValueError as e:
                    logger.warning(f"   ❌ Parse error: {e}")
            else:
                logger.info(f"   ❌ No match")
        
        # Strategy 2: Enhanced numbered list patterns  
        list_patterns = [
            r'\d+\.\s*(?:Image\s*)?(\d+)',                 # 1. Image 3 or 1. 3
            r'(\d+)\s*(?:is|comes|ranks?)\s*(?:first|1st|best)',  # 3 is first
            r'(?:first|1st|best):\s*(?:Image\s*)?(\d+)',   # First: Image 3
            r'(\d+)\s*-\s*(?:best|first)',                 # 3 - best
        ]
        
        logger.info("🔍 Testing numbered list patterns...")
        for i, pattern in enumerate(list_patterns, 1):
            logger.info(f"   List Pattern {i}: {pattern}")
            list_matches = re.findall(pattern, llm_response, re.IGNORECASE)
            logger.info(f"   Matches: {list_matches}")
            
            if len(list_matches) == num_images:
                try:
                    numbers = [int(x) for x in list_matches]
                    logger.info(f"   📊 Parsed numbers: {numbers}")
                    
                    if set(numbers) == set(range(1, num_images + 1)):
                        logger.info(f"✅ Valid ranking from list pattern {i}: {numbers}")
                        return numbers
                    else:
                        unique_numbers = set(numbers)
                        expected_set = set(range(1, num_images + 1))
                        has_duplicates = len(numbers) != len(unique_numbers)
                        
                        logger.warning(f"   ❌ Invalid list ranking:")
                        logger.warning(f"      - Numbers: {numbers}")
                        logger.warning(f"      - Has duplicates: {has_duplicates}")
                        logger.warning(f"      - Expected set: {expected_set}")
                        logger.warning(f"      - Actual set: {unique_numbers}")
                        
                        # Return the invalid ranking for quality validation if it has the right length
                        if len(numbers) == num_images:
                            logger.info(f"🔄 Returning invalid list ranking for quality validation: {numbers}")
                            return numbers
                except ValueError as e:
                    logger.warning(f"   ❌ Parse error: {e}")
        
        # Strategy 3: Look for any sequence of numbers
        logger.info("🔍 Looking for any sequence of numbers...")
        all_numbers = re.findall(r'\b(\d+)\b', llm_response)
        logger.info(f"   All numbers found: {all_numbers}")
        
        if all_numbers:
            try:
                # Filter numbers that are in valid range
                valid_numbers = [int(x) for x in all_numbers if 1 <= int(x) <= num_images]
                logger.info(f"   Valid range numbers: {valid_numbers}")
                
                # Check if we have exactly the right amount of unique numbers
                if len(set(valid_numbers)) == num_images and set(valid_numbers) == set(range(1, num_images + 1)):
                    # Take first occurrence of each number to preserve order
                    unique_ranking = []
                    seen = set()
                    for num in valid_numbers:
                        if num not in seen:
                            unique_ranking.append(num)
                            seen.add(num)
                    
                    if len(unique_ranking) == num_images:
                        logger.info(f"✅ Valid ranking from number sequence: {unique_ranking}")
                        return unique_ranking
                
                # If we have the right length but invalid content, return for quality check
                elif len(valid_numbers) == num_images:
                    logger.info(f"🔄 Returning sequence with right length for quality validation: {valid_numbers}")
                    return valid_numbers
                        
            except ValueError as e:
                logger.warning(f"   ❌ Parse error: {e}")
        
        # No valid ranking found
        logger.error("❌ ALL PARSING STRATEGIES FAILED")
        logger.error("📋 Response analysis:")
        logger.error(f"   - Length: {len(llm_response)} characters")
        logger.error(f"   - Contains brackets: {'[' in llm_response and ']' in llm_response}")
        
        # Fix f-string with backslash issue
        digit_pattern = r'\d+'
        contains_numbers = bool(re.search(digit_pattern, llm_response))
        logger.error(f"   - Contains numbers: {contains_numbers}")
        logger.error(f"   - Lines: {len(llm_response.splitlines())}")
        
        return None  # Signal that parsing failed
    
    def _validate_ranking_quality(self, ranking: List[int], expected_length: int) -> List[str]:
        """Validate ranking quality and return list of issues found."""
        issues = []
        
        # Check for correct length
        if len(ranking) != expected_length:
            issues.append(f"Wrong length: got {len(ranking)}, expected {expected_length}")
        
        # Check for duplicates
        unique_values = set(ranking)
        if len(unique_values) != len(ranking):
            duplicates = []
            seen = set()
            for num in ranking:
                if num in seen:
                    if num not in duplicates:
                        duplicates.append(num)
                seen.add(num)
            issues.append(f"Duplicates found: {duplicates}")
        
        # Check for missing numbers
        expected_set = set(range(1, expected_length + 1))
        actual_set = set(ranking)
        missing = expected_set - actual_set
        if missing:
            issues.append(f"Missing numbers: {sorted(missing)}")
        
        # Check for invalid numbers (outside range)
        invalid = actual_set - expected_set
        if invalid:
            issues.append(f"Invalid numbers (outside 1-{expected_length}): {sorted(invalid)}")
        
        # Check for zeros (common LLM mistake)
        if 0 in ranking:
            issues.append("Contains zero (invalid index)")
        
        # Check for numbers too large
        too_large = [x for x in ranking if x > expected_length]
        if too_large:
            issues.append(f"Numbers too large: {too_large}")
        
        return issues
    
    def get_llm_ranking_with_retry(self, caption: str, similar_images: List[Dict]) -> Tuple[List[int], str]:
        """Get LLM ranking with retry mechanism for invalid rankings."""
        voting_prompt = self.create_voting_prompt(caption, similar_images)
        logger.info(f"🤖 Asking {self.llm_model} to rank the images...")
        
        # Try with retry mechanism for invalid rankings
        ranking = None
        llm_response = None
        max_retry_attempts = 3
        original_prompt = voting_prompt  # Keep original prompt for retries
        
        for attempt in range(max_retry_attempts):
            logger.info(f"🎯 LLM Ranking Attempt {attempt + 1}/{max_retry_attempts}")
            
            # Get LLM response (use same prompt for retries when dealing with invalid rankings)
            llm_response = self.call_ollama_llm_with_retry(voting_prompt, max_retries=2)
            
            # Try to parse the ranking
            parsed_ranking = self.parse_llm_ranking(llm_response, len(similar_images))
            
            if parsed_ranking is not None:
                # Additional validation for specific ranking issues
                ranking_issues = self._validate_ranking_quality(parsed_ranking, len(similar_images))
                
                if not ranking_issues:
                    logger.info(f"✅ Successfully parsed valid ranking on attempt {attempt + 1}: {parsed_ranking}")
                    ranking = parsed_ranking
                    break
                else:
                    logger.warning(f"⚠️ Ranking has quality issues on attempt {attempt + 1}:")
                    for issue in ranking_issues:
                        logger.warning(f"   - {issue}")
                    
                    if attempt < max_retry_attempts - 1:
                        logger.info("🔄 Retrying with same prompt due to ranking issues...")
                        # Keep using the original prompt for these specific issues
                        voting_prompt = original_prompt
                    else:
                        logger.warning("⚠️ Using invalid ranking as last resort")
                        ranking = parsed_ranking
            else:
                logger.warning(f"⚠️ Failed to parse ranking on attempt {attempt + 1}")
                if attempt < max_retry_attempts - 1:
                    logger.info("🔄 Trying with modified prompt...")
                    # Only modify prompt for parsing failures, not ranking quality issues
                    if attempt == 0:
                        # First retry: add simple clarification
                        modified_prompt = original_prompt + f"\n\nIMPORTANT: Reply ONLY with numbers in brackets like this example: [1, 2, 3]\nDo not add any explanation or text.\n\nRanking:"
                    else:
                        # Second retry: be very explicit about duplicates
                        modified_prompt = original_prompt + f"""

CRITICAL: You MUST rank {len(similar_images)} images using numbers 1 to {len(similar_images)}.
FORBIDDEN examples:
- [1, 1, 2] ❌ (duplicate 1)
- [1, 3, 4] ❌ (missing 2)
- [1, 2, 3, 4] ❌ (wrong count)

CORRECT example for {len(similar_images)} images: {list(range(1, len(similar_images) + 1))}

Reply with ONLY the array: [x, y, z]

Ranking:"""
                    voting_prompt = modified_prompt
        
        # If all attempts failed, use fallback
        if ranking is None:
            logger.error("❌ All LLM ranking attempts failed, using CLIP order as fallback")
            ranking = list(range(1, len(similar_images) + 1))
            llm_response = f"FALLBACK: Could not parse LLM response after {max_retry_attempts} attempts"
        
        return ranking, llm_response
    
    def create_voting_prompt(self, target_caption: str, image_descriptions: List[Dict]) -> str:
        prompt = f"""You are an expert at matching text descriptions to images. 

Target caption to match: "{target_caption}"

Here are descriptions of {len(image_descriptions)} candidate images:

"""
        
        for i, img_info in enumerate(image_descriptions, 1):
            prompt += f"{i}. {img_info['blip_description']}\n"
        
        prompt += f"""
Instructions:
1. Analyze how well each image description matches the target caption
2. Consider semantic similarity, objects mentioned, scene context, and actions
3. Rank the images from 1 (best match) to {len(image_descriptions)} (worst match)

CRITICAL RULES:
- You MUST use each number from 1 to {len(image_descriptions)} exactly ONCE
- NO duplicates allowed (like [1, 1, 2] is INVALID)
- NO missing numbers (like [1, 3, 4] when you need 1,2,3 is INVALID)

YOU MUST respond with ONLY the ranking array, nothing else.
Example format: [3, 1, 5, 2, 4]

Ranking:"""
        
        return prompt
    
    def evaluate_ground_truth_ranking(self, caption: str, ground_truth_embedding_id: int, k: int = 5) -> Dict:
        """Evaluate if LLM voting improves the ranking of the ground truth image."""
        logger.info(f"🎯 Ground Truth Evaluation")
        logger.info(f"📝 Query Caption: {caption}")
        logger.info(f"🎲 Ground Truth Embedding ID: {ground_truth_embedding_id}")
        logger.info("=" * 60)
        
        # Step 1: CLIP retrieval
        similar_images = self.retrieve_similar_images(caption, k)
        
        # Check if ground truth image is in top-k results
        ground_truth_clip_rank = None
        ground_truth_found = False
        
        for result in similar_images:
            if result['faiss_index'] == ground_truth_embedding_id:
                ground_truth_clip_rank = result['clip_rank']
                ground_truth_found = True
                logger.info(f"✅ Ground truth found at CLIP rank #{ground_truth_clip_rank}")
                break
        
        if not ground_truth_found:
            logger.warning(f"❌ Ground truth image not in top-{k} CLIP results")
            return {
                'ground_truth_found': False,
                'ground_truth_clip_rank': None,
                'ground_truth_llm_rank': None,
                'improvement': 0,
                'clip_results': similar_images,
                'llm_results': None
            }
        
        # If ground truth is already #1, no need for LLM voting
        if ground_truth_clip_rank == 1:
            logger.info(f"✅ Ground truth already at rank #1, no LLM voting needed")
            return {
                'ground_truth_found': True,
                'ground_truth_clip_rank': 1,
                'ground_truth_llm_rank': 1,
                'improvement': 0,
                'clip_results': similar_images,
                'llm_results': None,
                'llm_response': None
            }
        
        # Step 2: Generate BLIP descriptions
        logger.info("🖼️ Generating BLIP descriptions...")
        for i, img_info in enumerate(similar_images):
            image_path = self.images_dir / img_info['filename']
            if image_path.exists():
                blip_caption = self.generate_blip_caption(image_path)
                img_info['blip_description'] = blip_caption
                img_info['image_path'] = str(image_path)
                logger.info(f"   {img_info['temp_index']}. {img_info['filename']}: {blip_caption[:60]}...")
            else:
                logger.warning(f"⚠️ Image not found: {image_path}")
                img_info['blip_description'] = "Image file not found"
                img_info['image_path'] = None
        
        # Step 3: LLM voting with retry mechanism for invalid rankings
        voting_prompt = self.create_voting_prompt(caption, similar_images)
        logger.info(f"🤖 Asking {self.llm_model} to rank the images...")
        
        # Try with retry mechanism for invalid rankings
        ranking = None
        llm_response = None
        max_retry_attempts = 3
        original_prompt = voting_prompt  # Keep original prompt for retries
        
        for attempt in range(max_retry_attempts):
            logger.info(f"🎯 LLM Ranking Attempt {attempt + 1}/{max_retry_attempts}")
            
            # Get LLM response (use same prompt for retries when dealing with invalid rankings)
            llm_response = self.call_ollama_llm_with_retry(voting_prompt, max_retries=2)
            
            # Try to parse the ranking
            parsed_ranking = self.parse_llm_ranking(llm_response, len(similar_images))
            
            if parsed_ranking is not None:
                # Additional validation for specific ranking issues
                ranking_issues = self._validate_ranking_quality(parsed_ranking, len(similar_images))
                
                if not ranking_issues:
                    logger.info(f"✅ Successfully parsed valid ranking on attempt {attempt + 1}: {parsed_ranking}")
                    ranking = parsed_ranking
                    break
                else:
                    logger.warning(f"⚠️ Ranking has quality issues on attempt {attempt + 1}:")
                    for issue in ranking_issues:
                        logger.warning(f"   - {issue}")
                    
                    if attempt < max_retry_attempts - 1:
                        logger.info("🔄 Retrying with same prompt due to ranking issues...")
                        # Keep using the original prompt for these specific issues
                        voting_prompt = original_prompt
                    else:
                        logger.warning("⚠️ Using invalid ranking as last resort")
                        ranking = parsed_ranking
            else:
                logger.warning(f"⚠️ Failed to parse ranking on attempt {attempt + 1}")
                if attempt < max_retry_attempts - 1:
                    logger.info("🔄 Trying with modified prompt...")
                    # Only modify prompt for parsing failures, not ranking quality issues
                    if attempt == 0:
                        # First retry: add simple clarification
                        modified_prompt = original_prompt + f"\n\nIMPORTANT: Reply ONLY with numbers in brackets like this example: [1, 2, 3]\nDo not add any explanation or text.\n\nRanking:"
                    else:
                        # Second retry: be very explicit about duplicates
                        modified_prompt = original_prompt + f"""

CRITICAL: You MUST rank {len(similar_images)} images using numbers 1 to {len(similar_images)}.
FORBIDDEN examples:
- [1, 1, 2] ❌ (duplicate 1)
- [1, 3, 4] ❌ (missing 2)
- [1, 2, 3, 4] ❌ (wrong count)

CORRECT example for {len(similar_images)} images: {list(range(1, len(similar_images) + 1))}

Reply with ONLY the array: [x, y, z]

Ranking:"""
                    voting_prompt = modified_prompt
        
        # If all attempts failed, use fallback
        if ranking is None:
            logger.error("❌ All LLM ranking attempts failed, using CLIP order as fallback")
            ranking = list(range(1, len(similar_images) + 1))
            llm_response = f"FALLBACK: Could not parse LLM response after {max_retry_attempts} attempts"
        
        # Step 4: Create LLM-ranked results
        llm_ranked_results = []
        for llm_rank, temp_idx in enumerate(ranking, 1):
            # Find the result with this temp_index
            original_result = None
            for result in similar_images:
                if result['temp_index'] == temp_idx:
                    original_result = result.copy()
                    break
            
            if original_result:
                original_result['llm_rank'] = llm_rank
                llm_ranked_results.append(original_result)
        
        # Find ground truth LLM rank
        ground_truth_llm_rank = None
        for result in llm_ranked_results:
            if result['faiss_index'] == ground_truth_embedding_id:
                ground_truth_llm_rank = result['llm_rank']
                break
        
        improvement = ground_truth_clip_rank - ground_truth_llm_rank
        
        return {
            'ground_truth_found': True,
            'ground_truth_clip_rank': ground_truth_clip_rank,
            'ground_truth_llm_rank': ground_truth_llm_rank,
            'improvement': improvement,
            'clip_results': similar_images,
            'llm_results': llm_ranked_results,
            'llm_response': llm_response
        }
    
    def display_evaluation_results(self, evaluation_result: Dict, query_caption: str):
        """Display the evaluation results comparing CLIP vs LLM ranking."""
        print("\n" + "=" * 80)
        print("🎯 GROUND TRUTH EVALUATION")
        print("=" * 80)
        print(f"📝 Query Caption: {query_caption}")
        print(f"🎲 Ground Truth Found: {'✅ Yes' if evaluation_result['ground_truth_found'] else '❌ No'}")
        
        if not evaluation_result['ground_truth_found']:
            print(f"❌ Ground truth image not in top-5 CLIP results")
            return
        
        clip_rank = evaluation_result['ground_truth_clip_rank']
        llm_rank = evaluation_result['ground_truth_llm_rank']
        improvement = evaluation_result['improvement']
        
        print(f"📊 Ground Truth Rankings:")
        print(f"   🔍 CLIP Rank: #{clip_rank}")
        print(f"   🤖 LLM Rank: #{llm_rank}")
        print(f"   📈 Improvement: {'+' if improvement > 0 else ''}{improvement} positions")
        
        if improvement > 0:
            print(f"🎉 LLM IMPROVED the ranking by {improvement} positions!")
        elif improvement < 0:
            print(f"⚠️ LLM made ranking worse by {abs(improvement)} positions")
        else:
            print(f"🔄 No change in ranking")
        
        print("=" * 80)
        print("📊 DETAILED RESULTS COMPARISON:")
        print("=" * 80)
        
        # Show CLIP results
        print(f"\n🔍 CLIP SIMILARITY RANKING:")
        clip_results = evaluation_result['clip_results']
        for result in clip_results:
            is_gt = "🎯 GT " if result['faiss_index'] == evaluation_result.get('ground_truth_embedding_id') else "    "
            print(f"{is_gt}#{result['clip_rank']}: {result['filename']} (sim: {result['similarity']:.4f})")
        
        # Show LLM results if available
        if evaluation_result['llm_results']:
            print(f"\n🤖 LLM RANKING (based on BLIP descriptions):")
            llm_results = evaluation_result['llm_results']
            for result in llm_results:
                is_gt = "🎯 GT " if result['faiss_index'] == evaluation_result.get('ground_truth_embedding_id') else "    "
                clip_pos = result['clip_rank']
                llm_pos = result['llm_rank']
                change = clip_pos - llm_pos
                change_str = f"({'+' if change > 0 else ''}{change})" if change != 0 else "(=)"
                print(f"{is_gt}#{llm_pos}: {result['filename']} {change_str}")
                print(f"      BLIP: {result.get('blip_description', 'N/A')[:70]}...")
        else:
            print(f"\n🤖 LLM RANKING (based on BLIP descriptions):")
            print("   (No LLM ranking needed - ground truth already at position #1)")
        
        print("=" * 80)
        
        # Special message when ground truth is already top 1
        if clip_rank == 1:
            print(f"🎯 Top 1 retrieved image is the corresponding one")

def main():
    parser = argparse.ArgumentParser(description="Ground Truth LLM Voting Image Retrieval")
    parser.add_argument('--vectordb', required=True, help='VectorDB name (e.g., Flickr_VectorDB)')
    parser.add_argument('--llm', default=None, help='LLM model (default: from Config)')
    parser.add_argument('--caption', type=str, help='Specific caption to test (optional)')
    parser.add_argument('--filename', type=str, help='Specific filename to get caption for (optional)')
    parser.add_argument('--random', action='store_true', help='Use random ground truth caption')
    parser.add_argument('--k', type=int, default=5, help='Number of results')
    parser.add_argument('--vectordb-dir', default=None, help='VectorDB directory (default: from Config)')
    
    args = parser.parse_args()
    
    try:
        retriever = GroundTruthLLMVotingRetriever(
            vectordb_name=args.vectordb,
            vectordb_dir=args.vectordb_dir,
            llm_model=args.llm
        )
        
        # Determine which caption and ground truth to use
        if args.filename:
            # Use specific filename
            result = retriever.get_caption_by_filename(args.filename)
            if not result:
                return
            caption, ground_truth_id, metadata = result
        elif args.caption:
            # User provided caption - need to find corresponding ground truth
            logger.error("❌ Custom caption mode not implemented. Use --filename or --random instead.")
            return
        else:
            # Use random ground truth
            caption, ground_truth_id, metadata = retriever.get_random_ground_truth_caption()
        
        # Run evaluation
        evaluation_result = retriever.evaluate_ground_truth_ranking(caption, ground_truth_id, args.k)
        
        # Add ground truth embedding id for display
        evaluation_result['ground_truth_embedding_id'] = ground_truth_id
        
        # Display results
        retriever.display_evaluation_results(evaluation_result, caption)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()