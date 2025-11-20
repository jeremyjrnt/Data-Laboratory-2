# Quick Reference - Centralized Configuration

## Import

```python
from config.config import Config, config
```

## URLs Ollama

```python
# Recommended method
Config.get_ollama_url('local')   # http://localhost:11434/api/generate
Config.get_ollama_url('a5000')   # http://100.64.0.7:11434/api/generate  
Config.get_ollama_url('a6000')   # http://100.64.0.9:11434/api/generate

# Direct access
Config.OLLAMA_LOCAL
Config.OLLAMA_REMOTE_A5000
Config.OLLAMA_REMOTE_A6000
```

## LLM Models

```python
# Recommended method
Config.get_ollama_model('default')   # gemma3:4b
Config.get_ollama_model('large')     # gemma3:27b
Config.get_ollama_model('mistral')   # mistral:7b

# Direct access
Config.OLLAMA_MODEL_DEFAULT
Config.OLLAMA_MODEL_LARGE
Config.OLLAMA_MODEL_MISTRAL
```

## HuggingFace Models

```python
Config.HF_MODEL_BLIP                 # Salesforce/blip-image-captioning-large
Config.HF_MODEL_CLIP                 # openai/clip-vit-base-patch32
Config.HF_MODEL_SENTENCE_TRANSFORMER # sentence-transformers/all-MiniLM-L6-v2
Config.HF_TOKEN                      # Your HF token
```

## Paths - Datasets

```python
# Recommended method (auto-constructs paths)
Config.get_dataset_dir('COCO')           # .../data/COCO
Config.get_images_dir('Flickr')          # .../data/Flickr/images
Config.get_metadata_path('VizWiz')       # .../data/VizWiz/vizwiz_metadata.json
Config.get_selected_images_path('COCO')  # .../data/COCO/selected_1000.json

# Direct access to directories
Config.DATA_DIR          # Path('.../data')
Config.COCO_DATA_DIR     # Path('.../data/COCO')
Config.FLICKR_DATA_DIR   # Path('.../data/Flickr')
Config.VIZWIZ_DATA_DIR   # Path('.../data/VizWiz')

Config.COCO_IMAGES_DIR   # Path('.../data/COCO/images')
Config.FLICKR_IMAGES_DIR # Path('.../data/Flickr/images')
Config.VIZWIZ_IMAGES_DIR # Path('.../data/VizWiz/images')
```

## Paths - VectorDB

```python
# Recommended method
Config.get_vectordb_index_path('COCO_VectorDB')     # .../VectorDBs/COCO_VectorDB.index
Config.get_vectordb_metadata_path('COCO_VectorDB') # .../VectorDBs/COCO_VectorDB_metadata.json

# Direct access
Config.VECTORDB_DIR  # Path('.../VectorDBs')
```

## Paths - Other

```python
Config.REPORT_DIR   # Path('.../report')
Config.CONFIG_DIR   # Path('.../config')
Config.PROJECT_ROOT # Path('.../DataLab2Project')
```

## Standard Filenames

```python
Config.METADATA_FILENAME           # "metadata.json"
Config.SELECTED_IMAGES_FILENAME    # "selected_1000.json"
Config.PERFORMANCE_FILENAME        # "performance.json"
Config.CLUSTER_POSITIONS_FILENAME  # "cluster_positions.json"
Config.CORPUS_FILENAME             # "corpus.json"
```

## Processing Parameters

```python
Config.CHUNK_SIZE        # 512
Config.CHUNK_OVERLAP     # 50
Config.MAX_TOKENS        # 2048
Config.MAX_NEW_TOKENS    # 40
```

## BLIP Parameters

```python
Config.BLIP_NUM_BEAMS          # 5
Config.BLIP_TOP_K              # 50
Config.BLIP_TOP_P              # 0.95
Config.BLIP_TEMPERATURE        # 0.8
Config.BLIP_REPETITION_PENALTY # 1.5
Config.BLIP_MIN_LENGTH         # 30
Config.BLIP_MAX_LENGTH         # 150

# Usage in generate()
out = model.generate(
    **inputs,
    max_new_tokens=Config.MAX_NEW_TOKENS,
    num_beams=Config.BLIP_NUM_BEAMS,
    top_k=Config.BLIP_TOP_K,
    top_p=Config.BLIP_TOP_P,
    temperature=Config.BLIP_TEMPERATURE,
    repetition_penalty=Config.BLIP_REPETITION_PENALTY,
    min_length=Config.BLIP_MIN_LENGTH,
    max_length=Config.BLIP_MAX_LENGTH
)

# Or use the helper method
params = Config.get_blip_generation_params()
out = model.generate(**inputs, **params)
```

## Retrieval Parameters

```python
Config.DEFAULT_K              # 100 - default number of results
Config.DEFAULT_TOP_K          # 10 - top-k results to return
Config.DEFAULT_THRESHOLD      # 0.0 - similarity threshold
Config.MAX_RETRY_ATTEMPTS     # 3 - max retry attempts for API calls
```

## FAISS IVF Settings

```python
Config.FAISS_NPROBE                # 10 - number of clusters to probe
Config.IVF_CLUSTER_CALCULATION     # 'sqrt' - calculation method
Config.IVF_ALPHA                   # 0.5 - alpha for weighting
```

## Rocchio PRF Parameters

```python
Config.ROCCHIO_ALPHA          # 1.0 - original query weight
Config.ROCCHIO_BETA           # 0.75 - relevant documents weight
Config.ROCCHIO_GAMMA          # 0.0 - non-relevant documents weight

# Usage
from config.config import Config

retriever = PRFRetriever(
    alpha=Config.ROCCHIO_ALPHA,
    beta=Config.ROCCHIO_BETA,
    gamma=Config.ROCCHIO_GAMMA
)

# Or use the helper method
params = Config.get_rocchio_params()
retriever = PRFRetriever(**params)
```

## Fusion Weights (Two Embeddings)

```python
Config.WEIGHT_CLASSIC         # 0.5 - classic embedding weight
Config.WEIGHT_BLIP            # 0.5 - BLIP embedding weight
Config.WEIGHT_AVERAGE         # 0.5 - average weight
Config.WEIGHT_BM25            # 0.5 - BM25 weight

# Usage
weights = Config.get_fusion_weights()
combined_score = (
    weights['classic'] * classic_score +
    weights['blip'] * blip_score
)
```

## Replacement Patterns

| Old (hardcoded) | New (Config) |
|---------------------|------------------|
| `"http://100.64.0.9:11434/api/generate"` | `Config.get_ollama_url('a6000')` |
| `"gemma3:27b"` | `Config.get_ollama_model('large')` |
| `"Salesforce/blip-image-captioning-large"` | `Config.HF_MODEL_BLIP` |
| `Path("data/COCO")` | `Config.get_dataset_dir('COCO')` |
| `Path("data/COCO/images")` | `Config.get_images_dir('COCO')` |
| `Path("VectorDBs")` | `Config.VECTORDB_DIR` |
| `"selected_1000.json"` | `Config.SELECTED_IMAGES_FILENAME` |
| `max_new_tokens=40` | `max_new_tokens=Config.MAX_NEW_TOKENS` |
| `k=100` | `k=Config.DEFAULT_K` |
| `threshold=0.0` | `threshold=Config.DEFAULT_THRESHOLD` |
| `alpha=1.0, beta=0.75` | `**Config.get_rocchio_params()` |
| `nprobe=10` | `nprobe=Config.FAISS_NPROBE` |

## Configuration in Constructors

```python
# Before
class MyClass:
    def __init__(self, data_dir="data/COCO", model="gemma3:4b"):
        self.data_dir = Path(data_dir)
        self.model = model

# After
class MyClass:
    def __init__(self, dataset_name="COCO", model_type="default"):
        self.data_dir = Config.get_dataset_dir(dataset_name)
        self.model = Config.get_ollama_model(model_type)
```

## Verify Configuration

```bash
# Display all configuration
python config/config.py

# See usage examples
python config/examples.py
```

## Modify Configuration

1. Edit `.env` at project root
2. Restart your Python application

**Note:** Do not modify `.env.example` (it's a template)
