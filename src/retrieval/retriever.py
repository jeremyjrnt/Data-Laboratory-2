"""
Text-to-Image Retrieval System using CLIP and FAISS
Retrieve the most similar images from a vector database based on text descriptions
"""

import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

class TextToImageRetriever:
    def __init__(self, vectordb_path: str = None, model_name: str = None):
        """
        Initialize the Text-to-Image Retriever
        
        Args:
            vectordb_path: Path to the VectorDBs directory (default: from Config)
            model_name: CLIP model name (default: from Config)
        """
        self.vectordb_path = Path(vectordb_path).resolve() if vectordb_path else Config.VECTORDB_DIR
        self.model_name = model_name or Config.HF_MODEL_CLIP_LARGE
        
        # Initialize CLIP model and processor
        print(f"🔄 Loading CLIP model: {self.model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎯 Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # Set to evaluation mode and enable half precision if using GPU
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.half()  # Use FP16 for faster inference
        
        # Current database state
        self.current_db = None
        self.current_index = None
        self.current_metadata = None
        self.db_name = None
        
        print("✅ Text-to-Image Retriever initialized")
    
    def list_available_databases(self) -> List[str]:
        """List all available vector databases"""
        if not self.vectordb_path.exists():
            return []
        
        databases = []
        for file in self.vectordb_path.glob("*.index"):
            db_name = file.stem
            metadata_file = self.vectordb_path / f"{db_name}_metadata.json"
            if metadata_file.exists():
                databases.append(db_name)
        
        return databases
    
    def load_database(self, db_name: str) -> bool:
        """Load the specified database"""
        try:
            index_file = self.vectordb_path / f"{db_name}.index"
            metadata_file = self.vectordb_path / f"{db_name}_metadata.json"
            
            if not index_file.exists() or not metadata_file.exists():
                print(f"❌ Database files not found for: {db_name}")
                return False
            
            # Load FAISS index
            print(f"📥 Loading FAISS index: {index_file.name}")
            self.current_index = faiss.read_index(str(index_file))
            
            # Load metadata
            print(f"📥 Loading metadata: {metadata_file.name}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                # Extract the metadata list from the dictionary
                self.current_metadata = metadata_dict.get("metadata", metadata_dict)
            
            self.db_name = db_name
            print(f"✅ Database '{db_name}' loaded successfully")
            print(f"   📊 {len(self.current_metadata):,} entries")
            print(f"   🎯 Vector dimension: {self.current_index.d}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP model"""
        try:
            # Process text
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text embedding
            with torch.no_grad():
                if self.device.type == "cuda":
                    # Use half precision for GPU
                    with torch.cuda.amp.autocast():
                        text_features = self.model.get_text_features(**inputs)
                else:
                    text_features = self.model.get_text_features(**inputs)
                
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = text_features.cpu().numpy().astype(np.float32)
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            return None
    
    def search_similar_images(self, text_query: str, k: int = 5, threshold: float = 0.0, verbose: bool = True) -> List[Tuple[Dict, float]]:
        """
        Search for images similar to the text query
        
        Args:
            text_query: Text description to search for
            k: Number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            verbose: Whether to print detailed results
        
        Returns:
            List of (metadata_dict, similarity_score) tuples
        """
        if not self.current_index or not self.current_metadata:
            print("❌ No database loaded")
            return []
        
        if verbose:
            print(f"🔍 Searching for: '{text_query}'")
        
        # Encode text query
        query_embedding = self.encode_text(text_query)
        if query_embedding is None:
            return []
        
        # Search in FAISS index
        try:
            distances, indices = self.current_index.search(query_embedding, k)
            
            results = []
            
            if verbose:
                print(f"\n📊 Found {len(indices[0])} results:")
                print("-" * 80)
            
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                similarity = float(dist)  # Cosine similarity (higher = more similar)
                
                if similarity < threshold:
                    continue
                
                # Get metadata
                metadata = self.current_metadata[idx]
                results.append((metadata, similarity))
                
                if verbose:
                    print(f"{i+1}. Similarity: {similarity:.4f} | ID: {idx}")
                    print(f"   📁 File: {metadata.get('filename', 'N/A')}")
                    print(f"   📝 Caption: {metadata.get('caption', 'N/A')}")
                    print(f"   📏 Size: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")
                    print()
            
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def display_image_with_metadata(self, metadata: Dict, similarity: float = None):
        """Display an image with its metadata"""
        try:
            image_path = metadata.get('image_path', '')
            
            # Convert relative path to absolute if needed
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.path.dirname(self.vectordb_path), image_path)
            
            image_path = Path(image_path)
            
            if not image_path.exists():
                print(f"❌ Image file not found: {image_path}")
                print("📋 Metadata only:")
                self._print_metadata(metadata, similarity)
                return
            
            # Load and display image
            print(f"🖼️  Displaying image: {image_path.name}")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Display image
            img = Image.open(image_path)
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f"Image: {metadata.get('filename', 'N/A')}")
            
            # Display metadata as text
            ax2.axis('off')
            metadata_text = self._format_metadata_for_display(metadata, similarity)
            ax2.text(0.1, 0.9, metadata_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error displaying image: {e}")
            print("📋 Metadata only:")
            self._print_metadata(metadata, similarity)
    
    def _format_metadata_for_display(self, metadata: Dict, similarity: float = None) -> str:
        """Format metadata for display in matplotlib"""
        lines = ["📋 METADATA", "=" * 40]
        
        if similarity is not None:
            lines.append(f"🎯 Similarity Score: {similarity:.4f}")
            lines.append("")
        
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) > 60:
                # Wrap long strings
                value = value[:60] + "..."
            lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _print_metadata(self, metadata: Dict, similarity: float = None):
        """Print metadata to console"""
        print("📋 METADATA:")
        print("=" * 40)
        
        if similarity is not None:
            print(f"🎯 Similarity Score: {similarity:.4f}")
            print()
        
        for key, value in metadata.items():
            print(f"{key}: {value}")
        print("=" * 40)
    
    def interactive_search(self):
        """Interactive search interfacet"""
        print("\n" + "=" * 60)
        print("🔍 TEXT-TO-IMAGE RETRIEVAL SYSTEM")
        print("=" * 60)
        
        # Select database if none loaded
        if not self.db_name:
            databases = self.list_available_databases()
            
            if not databases:
                print("❌ No vector databases found")
                return
            
            print("\n🗂️  Available databases:")
            print("-" * 50)
            for i, db in enumerate(databases, 1):
                # Get database info
                index_file = self.vectordb_path / f"{db}.index"
                metadata_file = self.vectordb_path / f"{db}_metadata.json"
                
                size_mb = index_file.stat().st_size / (1024 * 1024)
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                    # Handle both old format (direct list) and new format (dict with metadata key)
                    if isinstance(metadata_dict, dict) and "metadata" in metadata_dict:
                        num_entries = len(metadata_dict["metadata"])
                    else:
                        num_entries = len(metadata_dict)
                
                print(f"{i}. {db}")
                print(f"   📊 {num_entries:,} entries | 💾 {size_mb:.1f} MB")
            
            print("-" * 50)
            
            while True:
                try:
                    choice = input(f"\nSelect database (1-{len(databases)}) or 'q' to quit: ").strip()
                    
                    if choice.lower() == 'q':
                        print("👋 Goodbye!")
                        return
                    
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(databases):
                        if self.load_database(databases[choice_idx]):
                            break
                        else:
                            return
                    else:
                        print(f"❌ Please enter a number between 1 and {len(databases)}")
                except ValueError:
                    print("❌ Please enter a valid number or 'q' to quit")
        
        print(f"\n🗂️  Active database: {self.db_name}")
        print("💡 Enter text descriptions to find the most similar image")
        print("💡 Commands: 'stats' for database info, 'quit' to exit")
        
        while True:
            print("\n" + "-" * 60)
            query = input("🔍 Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'stats':
                self._show_database_stats()
                continue
            
            if not query:
                print("❌ Please enter a search query")
                continue
            
            # Search and display the top result only
            print(f"\n🔍 Searching for: '{query}'")
            result = self.search_and_display_top_result(query, show_image=True)
            
            if not result:
                print("❌ No results found")
    
    def _show_database_stats(self):
        """Show database statistics"""
        if not self.current_index or not self.current_metadata:
            print("❌ No database loaded")
            return
        
        print(f"\n📊 Database '{self.db_name}' Statistics:")
        print("=" * 50)
        print(f"📈 Total entries: {len(self.current_metadata):,}")
        print(f"🎯 Vector dimension: {self.current_index.d}")
        print(f"💾 Index vectors: {self.current_index.ntotal:,}")
        
        # Sample metadata fields
        if self.current_metadata:
            sample_entry = self.current_metadata[0]
            print(f"🏷️  Available fields: {', '.join(sample_entry.keys())}")
        
        print("=" * 50)
    
    def search_and_display_top_result(self, text_query: str, show_image: bool = True) -> Optional[Tuple[Dict, float]]:
        """
        Quick search that returns and displays the top result
        
        Args:
            text_query: Text description to search for
            show_image: Whether to display the image
        
        Returns:
            (metadata_dict, similarity_score) of top result or None
        """
        results = self.search_similar_images(text_query, k=1, verbose=False)
        
        if not results:
            print("❌ No results found")
            return None
        
        top_result = results[0]
        metadata, similarity = top_result
        
        print(f"\n🏆 TOP RESULT (Similarity: {similarity:.4f})")
        print("=" * 60)
        print(f"📁 File: {metadata.get('filename', 'N/A')}")
        print(f"📝 Caption: {metadata.get('caption', 'N/A')}")
        print(f"📏 Size: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}")
        
        if show_image:
            self.display_image_with_metadata(metadata, similarity)
        else:
            self._print_metadata(metadata, similarity)
        
        return top_result


def main():
    """Main function for testing"""
    retriever = TextToImageRetriever()
    
    # Start interactive search
    retriever.interactive_search()


if __name__ == "__main__":
    main()
