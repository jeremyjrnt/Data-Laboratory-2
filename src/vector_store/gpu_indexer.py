"""
GPU-Optimized FAISS Indexer Script
Optimized for fast GPU-based embedding generation with memory management
"""

import sys
import time
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from vector_store.indexer import FAISSVectorIndexer
from config.config import Config

def check_gpu_requirements():
    """Check GPU availability and requirements"""
    print("🔍 GPU Requirements Check")
    print("-" * 30)
    
    # Force re-import and clear cache
    import torch
    if hasattr(torch.cuda, '_initialization_lock'):
        torch.cuda._lazy_init()
    
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    
    if not cuda_available:
        print("❌ CUDA not detected by PyTorch.")
        print("This might be due to:")
        print("  - PyTorch CPU-only version installed")
        print("  - CUDA drivers not installed")
        print("  - Environment variables not set")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        if memory_gb < 4.0:
            print(f"⚠️  Warning: GPU {i} has limited memory ({memory_gb:.1f}GB)")
            print("   Consider reducing batch size or using CPU")
    
    return True

def optimize_gpu_settings():
    """Optimize GPU settings for embedding generation"""
    if torch.cuda.is_available():
        print("\n⚡ GPU Optimizations")
        print("-" * 20)
        
        # Set memory growth to avoid fragmentation
        torch.cuda.empty_cache()
        print("✅ Cleared GPU cache")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("✅ Enabled cuDNN optimizations")
        
        # Set memory management
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        print("✅ Set memory fraction to 80%")
        
        return True
    return False

def create_gpu_optimized_index(args):
    """Create index with GPU optimizations"""
    print("\n🚀 GPU-Optimized Index Creation")
    print("=" * 40)
    
    # Check paths
    images_path = Path(args.images_folder)
    metadata_path = Path(args.metadata_json)
    
    if not images_path.exists():
        print(f"❌ Images folder not found: {images_path}")
        return False
    
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return False
    
    try:
        # Create indexer with GPU device explicitly
        print(f"🔧 Creating indexer '{args.index_name}' with GPU...")
        indexer = FAISSVectorIndexer(
            index_name=args.index_name,
            model_name=args.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Show initial GPU stats
        initial_stats = indexer.get_stats()
        if "gpu_memory_allocated" in initial_stats:
            print(f"📊 Initial GPU Memory: {initial_stats['gpu_memory_allocated']}")
        
        # Start timing
        start_time = time.time()
        
        # Create index with optimized batch size for GPU
        batch_size = args.batch_size
        if torch.cuda.is_available():
            # Adjust batch size based on GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 12:
                batch_size = min(64, args.batch_size)  # High-end GPU
            elif gpu_memory_gb >= 8:
                batch_size = min(32, args.batch_size)  # Mid-range GPU
            else:
                batch_size = min(16, args.batch_size)  # Lower-end GPU
            
            print(f"🎛️  Using batch size: {batch_size} (based on {gpu_memory_gb:.1f}GB GPU)")
        
        print("\n🔄 Creating FAISS index...")
        success = indexer.create_index(
            images_folder=str(images_path),
            metadata_json=str(metadata_path),
            batch_size=batch_size
        )
        
        if not success:
            print("❌ Failed to create index")
            return False
        
        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save index
        save_path = Path(args.save_dir)
        print(f"\n💾 Saving index to {save_path}...")
        indexer.save_index(str(save_path))
        
        # Final statistics
        final_stats = indexer.get_stats()
        print(f"\n📊 Index Creation Complete!")
        print(f"   ⏱️  Total time: {total_time:.1f} seconds")
        print(f"   🖼️  Total vectors: {final_stats.get('total_vectors', 0)}")
        print(f"   📐 Embedding dimension: {final_stats.get('embedding_dimension', 0)}")
        
        if "gpu_memory_allocated" in final_stats:
            print(f"   🎮 GPU memory used: {final_stats['gpu_memory_allocated']}")
            print(f"   🎮 GPU memory reserved: {final_stats['gpu_memory_reserved']}")
        
        # Calculate throughput
        if final_stats.get('total_vectors', 0) > 0:
            throughput = final_stats['total_vectors'] / total_time
            print(f"   ⚡ Throughput: {throughput:.1f} images/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return False
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cleaned up GPU memory")

def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized FAISS Indexer')
    parser.add_argument('--index-name', required=True, help='Name for the FAISS index')
    parser.add_argument('--images-folder', required=True, help='Path to images folder')
    parser.add_argument('--metadata-json', required=True, help='Path to metadata JSON')
    parser.add_argument('--save-dir', default=None, help='Directory to save index (default: from Config)')
    parser.add_argument('--model-name', default=None, help='CLIP model name (default: from Config)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (auto-adjusted for GPU)')
    
    args = parser.parse_args()
    
    print("🚀 GPU-Optimized FAISS Vector Indexer")
    print("=" * 50)
    
    # Check GPU
    has_gpu = check_gpu_requirements()
    
    if has_gpu:
        optimize_gpu_settings()
    
    # Create index
    success = create_gpu_optimized_index(args)
    
    if success:
        save_dir = args.save_dir or str(Config.VECTORDB_DIR)
        print("\n🎉 Index creation completed successfully!")
        print(f"📁 Index saved to: {Path(save_dir).absolute()}")
    else:
        print("\n💥 Index creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
