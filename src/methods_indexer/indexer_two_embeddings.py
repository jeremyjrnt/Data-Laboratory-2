"""
Two Embeddings Indexer for Flickr Dataset
Creates two vector databases:
1. Flickr_blip_caption_VectorDB: Text embeddings from BLIP captions only
2. Flickr_average_VectorDB: Average of image and text embeddings (normalized)

Uses CLIP openai/clip-vit-large-patch14 on GPU with optimized inference
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
from tqdm import tqdm


class TwoEmbeddingsIndexer:
    """
    Indexer that creates two vector databases:
    - One with text embeddings only (BLIP captions)
    - One with averaged image + text embeddings
    """
    
    def __init__(self, dataset_name='Flickr'):
        """
        Initialize the indexer for Flickr dataset
        
        Args:
            dataset_name: Name of the dataset (default: Flickr)
        """
        self.dataset_name = dataset_name
        self.model_name = Config.HF_MODEL_CLIP_LARGE
        self.dimension = 768  # CLIP ViT-L/14 embedding dimension
        
        # CLIP model and processor
        self.model = None
        self.processor = None
        self.device = None
        
        # Paths
        self.base_path = Config.PROJECT_ROOT
        self.data_path = Config.get_dataset_dir(dataset_name)
        self.images_path = Config.get_images_dir(dataset_name)
        self.vectordb_path = Config.VECTORDB_DIR
        
        # Input metadata path
        self.input_metadata_path = self.vectordb_path / f"{dataset_name}_VectorDB_metadata.json"
        
        # Output paths for BLIP caption VectorDB
        self.blip_index_path = self.vectordb_path / f"{dataset_name}_blip_caption_VectorDB.index"
        self.blip_metadata_path = self.vectordb_path / f"{dataset_name}_blip_caption_VectorDB_metadata.json"
        
        # Output paths for averaged VectorDB
        self.avg_index_path = self.vectordb_path / f"{dataset_name}_average_VectorDB.index"
        self.avg_metadata_path = self.vectordb_path / f"{dataset_name}_average_VectorDB_metadata.json"
        
        # Create VectorDBs directory if it doesn't exist
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.metadata = []
        self.text_embeddings = []
        self.image_embeddings = []
        
    def _load_clip_model(self):
        """Load CLIP model and processor on GPU"""
        try:
            print(f"Loading CLIP model: {self.model_name}...")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to GPU
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            
            # Enable optimizations
            if torch.cuda.is_available():
                print("✓ CUDA optimizations enabled")
                torch.backends.cudnn.benchmark = True
                
        except Exception as e:
            print(f"✗ Error loading CLIP model: {e}")
            raise
        
    def load_metadata(self):
        """Load metadata from existing Flickr_VectorDB_metadata.json"""
        print(f"\nLoading metadata from {self.input_metadata_path}...")
        
        if not self.input_metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.input_metadata_path}")
        
        with open(self.input_metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata list
        if 'metadata' in data:
            self.metadata = data['metadata']
        else:
            raise ValueError("No 'metadata' field found in JSON")
        
        print(f"✓ Loaded {len(self.metadata)} image metadata entries")
        
        # Verify all entries have blip_caption
        missing_caption_count = 0
        for item in self.metadata:
            if 'blip_caption' not in item:
                missing_caption_count += 1
        
        if missing_caption_count > 0:
            print(f"⚠ Warning: {missing_caption_count} entries missing 'blip_caption'")
        
    def generate_embeddings(self):
        """
        Generate both text and image embeddings using CLIP
        Optimized to compute text embeddings once for both VectorDBs
        """
        print(f"\nGenerating embeddings for {len(self.metadata)} images...")
        
        with torch.no_grad():
            for idx, item in enumerate(tqdm(self.metadata, desc="🔄 Processing", unit="img", colour="green")):
                # Get BLIP caption
                blip_caption = item.get('blip_caption', '')
                
                if not blip_caption:
                    print(f"\n⚠ Warning: No BLIP caption for item {idx}, skipping...")
                    continue
                
                # Get image filename
                image_filename = item.get('filename') or item.get('file_name') or item.get('image')
                
                if not image_filename:
                    print(f"\n⚠ Warning: No filename for item {idx}, skipping...")
                    continue
                
                image_path = self.images_path / image_filename
                
                # Check if image exists
                if not image_path.exists():
                    print(f"\n⚠ Warning: Image not found: {image_path}, skipping...")
                    continue
                
                try:
                    # Load and process image
                    image = Image.open(image_path).convert('RGB')
                    
                    # Process both image and text together (optimized)
                    inputs = self.processor(
                        text=[blip_caption],
                        images=image,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    
                    # Move inputs to GPU
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get image features
                    image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
                    
                    # Get text features
                    text_features = self.model.get_text_features(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    
                    # Normalize features for cosine similarity
                    image_features = F.normalize(image_features, p=2, dim=1)
                    text_features = F.normalize(text_features, p=2, dim=1)
                    
                    # Convert to numpy
                    image_emb = image_features.cpu().numpy().astype('float32')[0]
                    text_emb = text_features.cpu().numpy().astype('float32')[0]
                    
                    # Store embeddings
                    self.text_embeddings.append(text_emb)
                    self.image_embeddings.append(image_emb)
                    
                except Exception as e:
                    print(f"\n✗ Error processing {image_path}: {e}")
                    continue
        
        # Convert to numpy arrays
        self.text_embeddings = np.array(self.text_embeddings, dtype='float32')
        self.image_embeddings = np.array(self.image_embeddings, dtype='float32')
        
        print(f"\n✓ Generated {len(self.text_embeddings)} text embeddings")
        print(f"✓ Generated {len(self.image_embeddings)} image embeddings")
        print(f"✓ Embedding dimension: {self.dimension}")
        
    def create_blip_caption_vectordb(self):
        """
        Create VectorDB with BLIP caption text embeddings only
        """
        print(f"\n{'='*60}")
        print("Creating Flickr_blip_caption_VectorDB...")
        print('='*60)
        
        # Create FAISS index
        index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity for normalized vectors)
        
        # Add text embeddings to index
        index.add(self.text_embeddings)
        
        print(f"✓ Index created with {index.ntotal} vectors")
        
        # Save FAISS index
        print(f"Saving index to {self.blip_index_path}...")
        faiss.write_index(index, str(self.blip_index_path))
        
        # Create metadata with only relevant fields
        metadata_output = {
            'index_name': f'{self.dataset_name}_blip_caption_VectorDB',
            'model_name': self.model_name,
            'embedding_type': 'text_only',
            'text_source': 'blip_caption',
            'embedding_dim': self.dimension,
            'total_vectors': int(index.ntotal),
            'metadata': self.metadata[:len(self.text_embeddings)]  # Only include successfully processed items
        }
        
        # Save metadata
        print(f"Saving metadata to {self.blip_metadata_path}...")
        with open(self.blip_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Flickr_blip_caption_VectorDB created successfully!")
        print(f"  - Index: {self.blip_index_path}")
        print(f"  - Metadata: {self.blip_metadata_path}")
        
    def create_average_vectordb(self):
        """
        Create VectorDB with averaged (image + text) embeddings, normalized
        """
        print(f"\n{'='*60}")
        print("Creating Flickr_average_VectorDB...")
        print('='*60)
        
        # Calculate average embeddings
        print("Computing averaged embeddings...")
        averaged_embeddings = (self.image_embeddings + self.text_embeddings) / 2.0
        
        # Normalize the averaged embeddings
        print("Normalizing averaged embeddings...")
        norms = np.linalg.norm(averaged_embeddings, axis=1, keepdims=True)
        averaged_embeddings = averaged_embeddings / norms
        averaged_embeddings = averaged_embeddings.astype('float32')
        
        print(f"✓ Averaged embeddings shape: {averaged_embeddings.shape}")
        
        # Create FAISS index
        index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity for normalized vectors)
        
        # Add averaged embeddings to index
        index.add(averaged_embeddings)
        
        print(f"✓ Index created with {index.ntotal} vectors")
        
        # Save FAISS index
        print(f"Saving index to {self.avg_index_path}...")
        faiss.write_index(index, str(self.avg_index_path))
        
        # Create metadata
        metadata_output = {
            'index_name': f'{self.dataset_name}_average_VectorDB',
            'model_name': self.model_name,
            'embedding_type': 'averaged_image_text',
            'text_source': 'blip_caption',
            'normalization': 'L2_normalized',
            'embedding_dim': self.dimension,
            'total_vectors': int(index.ntotal),
            'metadata': self.metadata[:len(averaged_embeddings)]  # Only include successfully processed items
        }
        
        # Save metadata
        print(f"Saving metadata to {self.avg_metadata_path}...")
        with open(self.avg_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Flickr_average_VectorDB created successfully!")
        print(f"  - Index: {self.avg_index_path}")
        print(f"  - Metadata: {self.avg_metadata_path}")
        
    def build(self):
        """Main method to build both VectorDBs"""
        print(f"\n{'='*70}")
        print(f"Two Embeddings VectorDB Builder for {self.dataset_name}")
        print('='*70)
        
        # Load CLIP model
        self._load_clip_model()
        
        # Load metadata
        self.load_metadata()
        
        # Generate embeddings (optimized: compute once, use twice)
        self.generate_embeddings()
        
        # Create both VectorDBs
        self.create_blip_caption_vectordb()
        self.create_average_vectordb()
        
        print(f"\n{'='*70}")
        print("✓ Both VectorDBs successfully created!")
        print('='*70)
        print("\nSummary:")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Total vectors: {len(self.text_embeddings)}")
        print(f"  Dimension: {self.dimension}")
        print("\nCreated VectorDBs:")
        print(f"  1. {self.blip_index_path.name} (text only)")
        print(f"  2. {self.avg_index_path.name} (averaged image + text)")
        print('='*70)


def main():
    """Main function to build the two VectorDBs"""
    indexer = TwoEmbeddingsIndexer(dataset_name='Flickr')
    indexer.build()


if __name__ == "__main__":
    main()
