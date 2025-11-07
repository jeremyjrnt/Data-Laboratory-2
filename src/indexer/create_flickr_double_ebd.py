"""
Script to create double embedding VectorDB for Flickr dataset
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from indexer.indexer_two_embeddings import DoubleEmbeddingIndexer
import time

def main():
    print("="*80)
    print("CREATING FLICKR DOUBLE EMBEDDING VECTORDB")
    print("="*80)
    
    # Paths - Using absolute paths from project root
    project_root = Path(__file__).parent.parent.parent
    index_name = "Flickr_VectorDB_double_ebd"
    images_folder = str(project_root / "data" / "Flickr" / "images")
    metadata_json = str(project_root / "data" / "Flickr" / "flickr_metadata.json")
    save_dir = str(project_root / "VectorDBs")
    
    print(f"\nConfiguration:")
    print(f"  Index name: {index_name}")
    print(f"  Images folder: {images_folder}")
    print(f"  Metadata JSON: {metadata_json}")
    print(f"  Save directory: {save_dir}")
    
    # Verify paths exist
    print(f"\nVerifying paths...")
    images_path = Path(images_folder)
    metadata_path = Path(metadata_json)
    
    if not images_path.exists():
        print(f"  ❌ Images folder NOT FOUND: {images_path}")
        return 1
    else:
        print(f"  ✅ Images folder found: {images_path}")
    
    if not metadata_path.exists():
        print(f"  ❌ Metadata JSON NOT FOUND: {metadata_path}")
        return 1
    else:
        print(f"  ✅ Metadata JSON found: {metadata_path}")
    
    print("="*80)
    
    # Create indexer
    start_time = time.time()
    
    indexer = DoubleEmbeddingIndexer(
        index_name=index_name,
        clip_model_name="openai/clip-vit-large-patch14",
        blip_model_name="Salesforce/blip-image-captioning-base"
    )
    
    # Create index
    print("\n🚀 Starting index creation...\n")
    success = indexer.create_index(
        images_folder=images_folder,
        metadata_json=metadata_json
    )
    
    if success:
        # Save index
        print("\n💾 Saving index...\n")
        indexer.save_index(save_dir)
        
        # Print stats
        end_time = time.time()
        total_time = end_time - start_time
        
        stats = indexer.get_stats()
        print("\n" + "="*80)
        print("📊 FINAL STATISTICS")
        print("="*80)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  • {k}: {v}")
            else:
                print(f"• {key}: {value}")
        
        print(f"\n⏱️  Total processing time: {total_time:.1f} seconds")
        if stats.get('images_count', 0) > 0:
            throughput = stats['images_count'] / total_time
            print(f"⚡ Throughput: {throughput:.2f} images/second")
        
        print("\n" + "="*80)
        print("✅ INDEX CREATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Index saved to: {Path(save_dir).resolve() / index_name}")
        
    else:
        print("\n" + "="*80)
        print("❌ INDEX CREATION FAILED!")
        print("="*80)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
