"""
IVF K-Means Indexer for Vector Database
Creates an inverted file index using K-Means clustering with sqrt(N) clusters
"""

import argparse
import json
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import faiss
from config.config import Config
from config.config import Config


class IVFKMeansIndexer:
    """
    Indexer using Inverted File structure with K-Means clustering
    Number of clusters = sqrt(N) where N is the number of vectors
    """
    
    def __init__(self, dataset_name, n_init=10):
        """
        Initialize the IVF K-Means indexer
        
        Args:
            dataset_name: Name of the dataset (COCO, Flickr, VizWiz)
            n_init: Number of initializations for K-Means (default: 10)
        """
        self.dataset_name = dataset_name
        self.n_init = n_init
        self.model_name = Config.HF_MODEL_CLIP_LARGE
        self.dimension = 768  # CLIP ViT-L/14 embedding dimension
        
        # CLIP model and processor
        self.model = None
        self.processor = None
        
        # Paths
        self.base_path = Config.PROJECT_ROOT
        self.data_path = Config.get_dataset_dir(dataset_name)
        self.metadata_path = Config.get_metadata_path(dataset_name)
        self.images_path = Config.get_images_dir(dataset_name)
        self.vectordb_path = Config.VECTORDB_DIR
        
        # Output paths
        self.index_path = self.vectordb_path / f"{dataset_name}_IVF_KMeans.index"
        self.metadata_output_path = self.vectordb_path / f"{dataset_name}_IVF_KMeans_metadata.json"
        self.assignments_path = self.vectordb_path / f"{dataset_name}_IVF_KMeans_assignments.pkl"
        self.centroids_path = self.vectordb_path / f"{dataset_name}_IVF_KMeans_centroids.npy"
        
        # Create VectorDBs directory if it doesn't exist
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = []
        self.metadata = []
        self.kmeans = None
        self.n_clusters = None
        
    def _load_clip_model(self):
        """Load CLIP model and processor"""
        try:
            print(f"Loading CLIP model: {self.model_name}...")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
        
    def load_metadata(self):
        """Load metadata from JSON file"""
        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'images' in data:
            self.metadata = data['images']
        elif isinstance(data, list):
            self.metadata = data
        else:
            raise ValueError("Unexpected metadata format")
        
        print(f"Loaded {len(self.metadata)} image metadata entries")
        
    def generate_embeddings(self):
        """Generate CLIP embeddings for all images"""
        print(f"Generating embeddings for {len(self.metadata)} images...")
        
        for idx, item in enumerate(self.metadata):
            if idx % 100 == 0:
                print(f"Processing image {idx}/{len(self.metadata)}...")
            
            # Get image filename
            if 'file_name' in item:
                image_filename = item['file_name']
            elif 'filename' in item:
                image_filename = item['filename']
            elif 'image' in item:
                image_filename = item['image']
            else:
                print(f"Warning: No filename found for item {idx}")
                continue
            
            image_path = self.images_path / image_filename
            
            # Check if image exists
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Process image with CLIP
                with torch.no_grad():
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get image features
                    image_features = self.model.get_image_features(**inputs)
                    
                    # Normalize for cosine similarity
                    image_features = F.normalize(image_features, p=2, dim=1)
                    
                    # Convert to numpy
                    embedding = image_features.cpu().numpy().astype('float32')[0]
                    self.embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        self.embeddings = np.array(self.embeddings, dtype='float32')
        print(f"Generated {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
        
    def create_ivf_index(self):
        """Create IVF index using K-Means clustering"""
        n_vectors = len(self.embeddings)
        
        # Calculate number of clusters as sqrt(N)
        self.n_clusters = int(np.sqrt(n_vectors))
        print(f"\nCreating IVF index with {self.n_clusters} clusters (sqrt of {n_vectors})...")
        
        # Train K-Means clustering
        print(f"Training K-Means with n_init={self.n_init}...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=42,
            verbose=1
        )
        
        # Fit K-Means and get cluster assignments
        cluster_assignments = self.kmeans.fit_predict(self.embeddings)
        centroids = self.kmeans.cluster_centers_.astype('float32')
        
        print(f"K-Means training completed")
        print(f"Cluster assignments shape: {cluster_assignments.shape}")
        print(f"Centroids shape: {centroids.shape}")
        
        # Create FAISS IVF index
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.n_clusters)
        
        # Train the index with the embeddings
        print("Training FAISS IVF index...")
        index.train(self.embeddings)
        
        # Add vectors to the index
        print("Adding vectors to index...")
        index.add(self.embeddings)
        
        print(f"Index created with {index.ntotal} vectors")
        
        # Save cluster assignments
        print(f"Saving cluster assignments to {self.assignments_path}...")
        assignments_data = {
            'assignments': cluster_assignments,
            'n_clusters': self.n_clusters,
            'n_vectors': n_vectors,
            'inertia': self.kmeans.inertia_
        }
        with open(self.assignments_path, 'wb') as f:
            pickle.dump(assignments_data, f)
        
        # Save centroids
        print(f"Saving centroids to {self.centroids_path}...")
        np.save(self.centroids_path, centroids)
        
        # Save FAISS index
        print(f"Saving FAISS index to {self.index_path}...")
        faiss.write_index(index, str(self.index_path))
        
        # Save metadata
        metadata_info = {
            'dataset': self.dataset_name,
            'n_vectors': n_vectors,
            'n_clusters': self.n_clusters,
            'n_init': self.n_init,
            'dimension': self.dimension,
            'index_type': 'IVFFlat',
            'clustering_method': 'KMeans',
            'inertia': float(self.kmeans.inertia_),
            'model': self.model_name
        }
        
        print(f"Saving metadata to {self.metadata_output_path}...")
        with open(self.metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_info, f, indent=2)
        
        print("\n" + "="*60)
        print("IVF K-Means Index Creation Summary")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Number of vectors: {n_vectors}")
        print(f"Number of clusters (√N): {self.n_clusters}")
        print(f"K-Means n_init: {self.n_init}")
        print(f"K-Means inertia: {self.kmeans.inertia_:.2f}")
        print(f"Vector dimension: {self.dimension}")
        print(f"\nFiles created:")
        print(f"  - Index: {self.index_path}")
        print(f"  - Metadata: {self.metadata_output_path}")
        print(f"  - Assignments: {self.assignments_path}")
        print(f"  - Centroids: {self.centroids_path}")
        print("="*60)
        
    def build(self):
        """Main method to build the IVF K-Means index"""
        print(f"\n{'='*60}")
        print(f"Building IVF K-Means Index for {self.dataset_name}")
        print(f"{'='*60}\n")
        
        self._load_clip_model()
        self.load_metadata()
        self.generate_embeddings()
        self.create_ivf_index()
        
        print(f"\n✓ IVF K-Means index successfully created for {self.dataset_name}!")


def main():
    parser = argparse.ArgumentParser(
        description='Create IVF K-Means index for image retrieval'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['COCO', 'Flickr', 'VizWiz'],
        help='Dataset name (COCO, Flickr, or VizWiz)'
    )
    parser.add_argument(
        '--n_init',
        type=int,
        default=10,
        help='Number of K-Means initializations (default: 10)'
    )
    
    args = parser.parse_args()
    
    indexer = IVFKMeansIndexer(args.dataset, n_init=args.n_init)
    indexer.build()


if __name__ == "__main__":
    main()
