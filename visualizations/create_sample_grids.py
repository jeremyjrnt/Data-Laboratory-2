"""
Script to create 3x3 grids of sampled images with their captions
for each dataset (COCO, Flickr, VizWiz)
"""

import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import textwrap
import time

# Paths
BASE_DIR = Path(r"C:\Users\binbi\Desktop\DataLab2Project")
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset configuration
DATASETS = {
    "COCO": {
        "metadata": BASE_DIR / "data" / "COCO" / "coco_metadata.json",
        "images_dir": BASE_DIR / "data" / "COCO" / "images",
        "caption_field": "caption"
    },
    "Flickr": {
        "metadata": BASE_DIR / "data" / "Flickr" / "flickr_metadata.json",
        "images_dir": BASE_DIR / "data" / "Flickr" / "images",
        "caption_field": "caption"
    },
    "VizWiz": {
        "metadata": BASE_DIR / "data" / "VizWiz" / "VizWiz_metadata.json",
        "images_dir": BASE_DIR / "data" / "VizWiz" / "images",
        "caption_field": "caption"
    }
}

def load_metadata(metadata_path):
    """Load metadata JSON file"""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['images']

def sample_random_images(images_list, n=9):
    """Select n random images"""
    return random.sample(images_list, min(n, len(images_list)))

def wrap_text(text, width=40):
    """Wrap text with line breaks"""
    return '\n'.join(textwrap.wrap(text, width=width))

def create_grid(dataset_name, config):
    """Create a 3x3 grid for a dataset with random sampling each time"""
    # Use current time for random seed to get different samples each run
    random.seed(time.time())
    
    print(f"Processing {dataset_name}...")
    
    # Load metadata
    images_metadata = load_metadata(config['metadata'])
    
    # Filter for short captions if specified for this dataset
    if config.get('max_caption_words'):
        max_words = config['max_caption_words']
        caption_field = config['caption_field']
        filtered = [
            img for img in images_metadata 
            if len(img.get(caption_field, '').split()) <= max_words
        ]
        print(f"  Filtered to {len(filtered)} images with captions ≤ {max_words} words")
        images_metadata = filtered
    
    # Sample 9 random images
    sampled = sample_random_images(images_metadata, 9)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 16))
    fig.suptitle(f'{dataset_name} Dataset - Sample Images with Captions', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(sampled):
            img_info = sampled[idx]
            img_filename = img_info['filename']
            caption = img_info.get(config['caption_field'], img_info.get('caption', 'No caption available'))
            
            # Load image
            img_path = config['images_dir'] / img_filename
            
            try:
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
                
                # Add caption below image
                wrapped_caption = wrap_text(caption, width=35)
                ax.text(0.5, -0.05, wrapped_caption, 
                       transform=ax.transAxes,
                       fontsize=9,
                       ha='center',
                       va='top',
                       wrap=True,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
            except Exception as e:
                print(f"Error loading {img_filename}: {e}")
                ax.text(0.5, 0.5, f'Error:\n{img_filename}', 
                       ha='center', va='center',
                       transform=ax.transAxes)
                ax.axis('off')
        else:
            ax.axis('off')
    
    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_path = OUTPUT_DIR / f"{dataset_name}_sample_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grid saved: {output_path}")
    
    plt.close()

def main():
    """Main function"""
    print("Creating sample grids for each dataset...\n")
    
    for dataset_name, config in DATASETS.items():
        try:
            create_grid(dataset_name, config)
            print(f"✓ {dataset_name} completed\n")
        except Exception as e:
            print(f"✗ Error for {dataset_name}: {e}\n")
    
    print("All grids have been created successfully!")
    print(f"Files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
