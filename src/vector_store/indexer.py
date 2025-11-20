"""
FAISS Vector Store Indexer with CLIP Embeddings

This module creates and manages FAISS indices for image embeddings using CLIP model.
Supports cosine similarity for image search and retrieval.

Features:
- Initialize FAISS index with custom name
- Process images from folder with metadata JSON
- Use OpenAI CLIP model for embeddings
- Cosine similarity scoring
-            # Process text query
            inputs = self.processor(text=text_query, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Convert to appropriate dtype for GPU
            if self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            # Search in FAISS index
            query_vector = text_features.cpu().float().numpy().astype('float32') index functionality
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm

# Core libraries
import faiss
from PIL import Image
import torch
import torch.nn.functional as F

# Transformers for CLIP
from transformers import CLIPProcessor, CLIPModel

# Config
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSVectorIndexer:
    """
    FAISS Vector Store for image embeddings using CLIP model
    """
    
    def __init__(self, 
                 index_name: str,
                 model_name: str = None,
                 device: Optional[str] = None):
        """
        Initialize FAISS indexer with CLIP model
        
        Args:
            index_name: Name for the FAISS index
            model_name: HuggingFace model name for CLIP
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.index_name = index_name
        self.model_name = model_name or Config.HF_MODEL_CLIP_LARGE
        
        # Set device with GPU priority
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.device = "cpu"
                logger.warning("⚠️  No GPU available, falling back to CPU")
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and processor
        self._load_clip_model()
        
        # FAISS index and metadata
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        
        logger.info(f"Initialized FAISS Indexer '{index_name}' with model '{model_name}'")
    
    def _load_clip_model(self):
        """Load CLIP model and processor"""
        try:
            logger.info(f"Loading CLIP model: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to device and optimize for inference
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Enable GPU optimizations if available
            if self.device == "cuda":
                # Enable mixed precision for faster inference
                self.model = self.model.half()  # Use FP16 for faster GPU inference
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                logger.info("🚀 GPU optimizations enabled: FP16 precision, cuDNN benchmark")
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.vision_config.hidden_size
            logger.info(f"CLIP model loaded successfully. Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _process_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Process single image and return embedding
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding or None if failed
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Convert to appropriate dtype for GPU
            if self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize for cosine similarity
                image_features = F.normalize(image_features, p=2, dim=1)
            
            return image_features.cpu().float().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None
    
    def create_index(self, 
                    images_folder: str, 
                    metadata_json: str,
                    batch_size: int = 32) -> bool:
        """
        Create FAISS index from images folder and metadata
        
        Args:
            images_folder: Path to folder containing images
            metadata_json: Path to JSON file with metadata
            batch_size: Batch size for processing images (adjust based on GPU memory)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear GPU cache at start
            if self.device == "cuda":
                torch.cuda.empty_cache()
                logger.info("🧹 Cleared GPU cache")
            
            images_path = Path(images_folder)
            metadata_path = Path(metadata_json)
            
            # Validate inputs
            if not images_path.exists():
                logger.error(f"Images folder not found: {images_path}")
                return False
            
            if not metadata_path.exists():
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
            
            # Load metadata
            logger.info(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if 'images' not in metadata:
                logger.error("Metadata JSON must contain 'images' field")
                return False
            
            images_metadata = metadata['images']
            logger.info(f"Found {len(images_metadata)} images in metadata")
            
            # Process images in batches
            embeddings_list = []
            valid_metadata = []
            
            logger.info("Processing images and generating embeddings...")
            
            for img_meta in tqdm(images_metadata, desc="Processing images"):
                filename = img_meta.get('filename', '')
                image_path = images_path / filename
                
                # Check if image exists
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Generate embedding
                embedding = self._process_image(image_path)
                
                if embedding is not None:
                    embeddings_list.append(embedding)
                    valid_metadata.append({
                        **img_meta,
                        'image_path': str(image_path),
                        'embedding_id': len(embeddings_list) - 1
                    })
            
            if not embeddings_list:
                logger.error("No valid embeddings generated")
                return False
            
            # Initialize FAISS index AFTER we have embeddings (so we know the dimension)
            embedding_dim = embeddings_list[0].shape[0]
            self.embedding_dim = embedding_dim
            
            # Create FAISS index for cosine similarity
            # Using Inner Product for normalized vectors (equivalent to cosine similarity)
            self.index = faiss.IndexFlatIP(embedding_dim)
            logger.info(f"Created FAISS index with dimension {embedding_dim}")
            
            # Convert to numpy array and add to index
            embeddings_array = np.vstack(embeddings_list).astype('float32')
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            self.metadata = valid_metadata
            
            logger.info(f"Successfully created FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def save_index(self, save_dir: str) -> bool:
        """
        Save FAISS index and metadata to disk
        
        Args:
            save_dir: Directory to save index files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_file = save_path / f"{self.index_name}.index"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = save_path / f"{self.index_name}_metadata.json"
            index_info = {
                'index_name': self.index_name,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'total_vectors': self.index.ntotal,
                'metadata': self.metadata
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(index_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Index saved successfully to {save_path}")
            logger.info(f"  - Index file: {index_file}")
            logger.info(f"  - Metadata file: {metadata_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, load_dir: str) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Args:
            load_dir: Directory containing index files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            load_path = Path(load_dir)
            
            # Load FAISS index
            index_file = load_path / f"{self.index_name}.index"
            if not index_file.exists():
                logger.error(f"Index file not found: {index_file}")
                return False
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            metadata_file = load_path / f"{self.index_name}_metadata.json"
            if not metadata_file.exists():
                logger.error(f"Metadata file not found: {metadata_file}")
                return False
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                index_info = json.load(f)
            
            self.metadata = index_info.get('metadata', [])
            self.embedding_dim = index_info.get('embedding_dim', self.embedding_dim)
            
            logger.info(f"Index loaded successfully from {load_path}")
            logger.info(f"  - Total vectors: {self.index.ntotal}")
            logger.info(f"  - Embedding dimension: {self.embedding_dim}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search_similar_images(self, 
                            query_image_path: str, 
                            k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar images using cosine similarity
        
        Args:
            query_image_path: Path to query image
            k: Number of similar images to return
            
        Returns:
            List of similar images with metadata and scores
        """
        try:
            if self.index is None:
                logger.error("Index not initialized. Create or load an index first.")
                return []
            
            # Process query image
            query_embedding = self._process_image(Path(query_image_path))
            if query_embedding is None:
                logger.error(f"Failed to process query image: {query_image_path}")
                return []
            
            # Search in FAISS index
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            scores, indices = self.index.search(query_vector, k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):
                    result = {
                        'rank': i + 1,
                        'score': float(score),  # Cosine similarity score
                        'metadata': self.metadata[idx]
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar images for query: {query_image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar images: {e}")
            return []
    
    def search_by_text(self, 
                      text_query: str, 
                      k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for images using text query (CLIP text-image similarity)
        
        Args:
            text_query: Text description to search for
            k: Number of similar images to return
            
        Returns:
            List of similar images with metadata and scores
        """
        try:
            if self.index is None:
                logger.error("Index not initialized. Create or load an index first.")
                return []
            
            # Process text query
            inputs = self.processor(text=text_query, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            # Search in FAISS index
            query_vector = text_features.cpu().numpy().astype('float32')
            scores, indices = self.index.search(query_vector, k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):
                    result = {
                        'rank': i + 1,
                        'score': float(score),  # Cosine similarity score
                        'metadata': self.metadata[idx]
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar images for text query: '{text_query}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by text: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics including GPU info"""
        stats = {
            "index_name": self.index_name,
            "model_name": self.model_name,
            "device": self.device,
            "index_type": "FAISS IndexFlatIP (Cosine Similarity)"
        }
        
        if self.index is not None:
            stats.update({
                "total_vectors": self.index.ntotal,
                "embedding_dimension": self.embedding_dim,
            })
        else:
            stats["status"] = "No index loaded"
        
        # Add GPU info if available
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_stats = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB",
            }
            stats.update(gpu_stats)
        
        return stats

def main():
    """Example usage of FAISSVectorIndexer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FAISS Vector Store Indexer')
    parser.add_argument('--index-name', required=True, help='Name for the FAISS index')
    parser.add_argument('--images-folder', required=True, help='Path to images folder')
    parser.add_argument('--metadata-json', required=True, help='Path to metadata JSON file')
    parser.add_argument('--save-dir', default=None, help='Directory to save index (default: from Config)')
    parser.add_argument('--model-name', default=None, help='CLIP model name (default: from Config)')
    
    args = parser.parse_args()
    
    # Create indexer
    indexer = FAISSVectorIndexer(
        index_name=args.index_name,
        model_name=args.model_name
    )
    
    # Create index
    logger.info("Starting index creation...")
    success = indexer.create_index(
        images_folder=args.images_folder,
        metadata_json=args.metadata_json
    )
    
    if success:
        # Save index
        save_dir = args.save_dir or str(Config.VECTORDB_DIR)
        indexer.save_index(save_dir)
        
        # Print stats
        stats = indexer.get_stats()
        print("\n📊 Index Statistics:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
        
        logger.info("Index creation completed successfully!")
    else:
        logger.error("Index creation failed!")

if __name__ == "__main__":
    main()
