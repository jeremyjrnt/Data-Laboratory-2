"""
FAISS Vector Store Indexer with Double Embeddings (Image + BLIP Text)

This module creates a FAISS index with two embeddings per image:
1. CLIP embedding of the image itself
2. CLIP embedding of the BLIP-generated text description
3. Average of both embeddings (normalized)

Each vector in the database has metadata indicating its source type:
- 'image': Direct CLIP embedding of the image
- 'blip_text': CLIP embedding of BLIP-generated description
- 'average': Normalized average of image and text embeddings
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DoubleEmbeddingIndexer:
    """
    FAISS Vector Store with double embeddings (image + BLIP text) per image
    """
    
    def __init__(self,
                 index_name: str = "double_embedding",
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 blip_model_name: str = "Salesforce/blip-image-captioning-large",
                 device: Optional[str] = None):
        """
        Initialize Double Embedding Indexer
        
        Args:
            index_name: Name for the FAISS index
            clip_model_name: HuggingFace CLIP model name
            blip_model_name: HuggingFace BLIP model name
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.index_name = index_name
        self.clip_model_name = clip_model_name
        self.blip_model_name = blip_model_name
        
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
        
        # FAISS index and metadata
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        
        # Initialize models
        self._load_models()
        
        logger.info(f"Initialized Double Embedding Indexer '{index_name}'")
    
    def _load_models(self):
        """Load CLIP and BLIP models"""
        try:
            # Load CLIP model
            logger.info(f"Loading CLIP model: {self.clip_model_name}")
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Get embedding dimension from the model
            # CLIP ViT-L/14 has 768 dimensions
            # We'll determine this by actually getting an embedding
            logger.info("Determining embedding dimension...")
            
            # Test with a simple text to get actual dimension
            test_inputs = self.clip_processor(text=["test"], return_tensors="pt", padding=True)
            test_inputs = {k: v.to(self.device) for k, v in test_inputs.items()}
            
            with torch.no_grad():
                test_features = self.clip_model.get_text_features(**test_inputs)
                self.embedding_dim = int(test_features.shape[-1])  # Convert to Python int
            
            logger.info(f"CLIP model loaded. Embedding dim: {self.embedding_dim} (type: {type(self.embedding_dim).__name__})")
            
            # Load BLIP model
            logger.info(f"Loading BLIP model: {self.blip_model_name}")
            self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_model_name)
            
            self.blip_model = self.blip_model.to(self.device)
            self.blip_model.eval()
            
            logger.info("BLIP model loaded successfully")
            
            # Enable GPU optimizations if available
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                logger.info("GPU optimizations enabled")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def generate_blip_caption(self, image_path: str) -> str:
        """
        Generate caption for an image using BLIP
        
        Args:
            image_path: Path to the image
            
        Returns:
            Generated caption
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.blip_model.generate(
                    **inputs,
                    max_length=150,        # Longer descriptions
                    min_length=30,         # Ensure minimum detail
                    num_beams=8,           # Better search
                    no_repeat_ngram_size=3, # Avoid repetition
                    length_penalty=1.0,    # Balanced length
                    early_stopping=True,
                    do_sample=False        # Deterministic for consistency
                )
            
            caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Error generating BLIP caption for {image_path}: {e}")
            return ""
    
    def get_clip_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Get CLIP embedding for an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Image embedding or None if failed
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.cpu().float().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error getting CLIP image embedding for {image_path}: {e}")
            return None
    
    def get_clip_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get CLIP embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding or None if failed
        """
        try:
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            return text_features.cpu().float().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error getting CLIP text embedding: {e}")
            return None
    
    def create_index(self,
                    images_folder: str,
                    metadata_json: str,
                    max_images: Optional[int] = None) -> bool:
        """
        Create FAISS index with double embeddings for each image
        
        Args:
            images_folder: Path to folder containing images
            metadata_json: Path to JSON file with metadata
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear GPU cache at start
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
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
                logger.error("Metadata JSON must contain 'images' key")
                return False
            
            images_metadata = metadata['images']
            logger.info(f"Found {len(images_metadata)} images in metadata")
            
            # Limit number of images if specified
            if max_images is not None and max_images > 0:
                images_metadata = images_metadata[:max_images]
                logger.info(f"Limited to {len(images_metadata)} images for processing")
            
            # Process images and generate embeddings
            all_embeddings = []
            all_metadata = []
            
            logger.info("Processing images and generating double embeddings...")
            
            for img_meta in tqdm(images_metadata, desc="Processing images"):
                # Get image path from metadata
                img_path_from_meta = img_meta.get('image_path', '')
                
                # Try different path resolution strategies
                img_path = None
                
                # Strategy 1: Use path as-is if it exists
                if img_path_from_meta and Path(img_path_from_meta).exists():
                    img_path = img_path_from_meta
                # Strategy 2: Use filename from metadata and join with images_folder
                elif img_path_from_meta:
                    filename = os.path.basename(img_path_from_meta)
                    potential_path = images_path / filename
                    if potential_path.exists():
                        img_path = str(potential_path)
                # Strategy 3: Use image_id as filename
                if not img_path and 'image_id' in img_meta:
                    image_id = img_meta['image_id']
                    # Try common extensions
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        potential_path = images_path / f"{image_id}{ext}"
                        if potential_path.exists():
                            img_path = str(potential_path)
                            break
                
                if not img_path:
                    logger.warning(f"Image not found: {img_path_from_meta} (tried: {images_path / os.path.basename(img_path_from_meta) if img_path_from_meta else 'N/A'})")
                    continue
                
                # 1. Get CLIP image embedding
                image_embedding = self.get_clip_image_embedding(img_path)
                if image_embedding is None:
                    logger.warning(f"Failed to get image embedding for: {img_path}")
                    continue
                
                # 2. Generate BLIP caption
                blip_caption = self.generate_blip_caption(img_path)
                if not blip_caption:
                    logger.warning(f"Failed to generate BLIP caption for: {img_path}")
                    continue
                
                # 3. Get CLIP text embedding of BLIP caption
                text_embedding = self.get_clip_text_embedding(blip_caption)
                if text_embedding is None:
                    logger.warning(f"Failed to get text embedding for: {img_path}")
                    continue
                
                # 4. Calculate average embedding (normalized)
                average_embedding = (image_embedding + text_embedding) / 2.0
                # Normalize the average
                norm = np.linalg.norm(average_embedding)
                if norm > 0:
                    average_embedding = average_embedding / norm
                
                # Add all three embeddings to the index
                # Each embedding gets its own metadata entry
                
                # Image embedding
                all_embeddings.append(image_embedding)
                all_metadata.append({
                    'image_path': img_path,
                    'caption': img_meta.get('caption', ''),
                    'blip_caption': blip_caption,
                    'embedding_type': 'image',
                    'image_id': img_meta.get('image_id', os.path.basename(img_path))
                })
                
                # BLIP text embedding
                all_embeddings.append(text_embedding)
                all_metadata.append({
                    'image_path': img_path,
                    'caption': img_meta.get('caption', ''),
                    'blip_caption': blip_caption,
                    'embedding_type': 'blip_text',
                    'image_id': img_meta.get('image_id', os.path.basename(img_path))
                })
                
                # Average embedding
                all_embeddings.append(average_embedding)
                all_metadata.append({
                    'image_path': img_path,
                    'caption': img_meta.get('caption', ''),
                    'blip_caption': blip_caption,
                    'embedding_type': 'average',
                    'image_id': img_meta.get('image_id', os.path.basename(img_path))
                })
            
            if not all_embeddings:
                logger.error("No embeddings generated")
                return False
            
            # Create FAISS index for cosine similarity
            logger.info(f"Creating FAISS index with {len(all_embeddings)} vectors...")
            logger.info(f"Embedding dimension: {self.embedding_dim} (type: {type(self.embedding_dim)})")
            
            # Ensure it's a proper Python int
            dim = int(self.embedding_dim)
            logger.info(f"Converted dimension: {dim} (type: {type(dim)})")
            
            self.index = faiss.IndexFlatIP(dim)
            
            # Convert to numpy array and add to index
            embeddings_array = np.vstack(all_embeddings).astype('float32')
            self.index.add(embeddings_array)
            self.metadata = all_metadata
            
            # Calculate statistics
            num_images = len(images_metadata)
            num_successful = len(all_metadata) // 3  # Each image has 3 embeddings
            
            logger.info(f"Successfully created FAISS index:")
            logger.info(f"  - Total images processed: {num_images}")
            logger.info(f"  - Successfully embedded: {num_successful}")
            logger.info(f"  - Total vectors in index: {self.index.ntotal}")
            logger.info(f"  - Embeddings per image: 3 (image, blip_text, average)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Clean up GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
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
                'clip_model_name': self.clip_model_name,
                'blip_model_name': self.blip_model_name,
                'embedding_dim': self.embedding_dim,
                'total_vectors': self.index.ntotal,
                'embedding_types': ['image', 'blip_text', 'average'],
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
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        stats = {
            "index_name": self.index_name,
            "clip_model_name": self.clip_model_name,
            "blip_model_name": self.blip_model_name,
            "device": self.device,
            "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
            "embedding_types": ["image", "blip_text", "average"]
        }
        
        if self.index is not None:
            stats.update({
                "total_vectors": self.index.ntotal,
                "embedding_dimension": self.embedding_dim,
                "images_count": self.index.ntotal // 3,  # Each image has 3 embeddings
            })
            
            # Count by embedding type
            embedding_type_counts = {}
            for meta in self.metadata:
                emb_type = meta.get('embedding_type', 'unknown')
                embedding_type_counts[emb_type] = embedding_type_counts.get(emb_type, 0) + 1
            
            stats['embedding_type_counts'] = embedding_type_counts
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
    """Main function for double embedding indexer"""
    parser = argparse.ArgumentParser(
        description='FAISS Double Embedding Indexer (Image + BLIP Text)'
    )
    parser.add_argument(
        '--index-name',
        type=str,
        required=True,
        help='Name for the FAISS index'
    )
    parser.add_argument(
        '--images-folder',
        type=str,
        required=True,
        help='Path to images folder'
    )
    parser.add_argument(
        '--metadata-json',
        type=str,
        required=True,
        help='Path to metadata JSON file'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='../../VectorDBs',
        help='Directory to save index (default: ../../VectorDBs)'
    )
    parser.add_argument(
        '--clip-model',
        type=str,
        default='openai/clip-vit-large-patch14',
        help='CLIP model name (default: openai/clip-vit-large-patch14)'
    )
    parser.add_argument(
        '--blip-model',
        type=str,
        default='Salesforce/blip-image-captioning-large',
        help='BLIP model name (default: Salesforce/blip-image-captioning-large)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (default: None = all images)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FAISS DOUBLE EMBEDDING INDEXER")
    print("="*80)
    print(f"Index name: {args.index_name}")
    print(f"Images folder: {args.images_folder}")
    print(f"Metadata: {args.metadata_json}")
    print(f"Save directory: {args.save_dir}")
    print("="*80)
    
    # Create indexer
    start_time = time.time()
    
    indexer = DoubleEmbeddingIndexer(
        index_name=args.index_name,
        clip_model_name=args.clip_model,
        blip_model_name=args.blip_model
    )
    
    # Create index
    logger.info("Starting double embedding index creation...")
    success = indexer.create_index(
        images_folder=args.images_folder,
        metadata_json=args.metadata_json,
        max_images=args.max_images
    )
    
    if success:
        # Save index
        indexer.save_index(args.save_dir)
        
        # Print stats
        end_time = time.time()
        total_time = end_time - start_time
        
        stats = indexer.get_stats()
        print("\n" + "="*80)
        print("📊 INDEX STATISTICS")
        print("="*80)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  - {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print(f"\n⏱️  Total time: {total_time:.1f} seconds")
        if stats.get('images_count', 0) > 0:
            throughput = stats['images_count'] / total_time
            print(f"⚡ Throughput: {throughput:.2f} images/second")
        
        print("="*80)
        logger.info("Index creation completed successfully! ✅")
    else:
        logger.error("Index creation failed! ❌")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
