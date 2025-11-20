# Centralized Configuration - Summary

## ✅ Created Files

### 1. Main Configuration
- **`config/config.py`** - Main module with `Config` class and `config` singleton
- **`config/__init__.py`** - Package initialization
- **`.env`** - Configuration file (updated with all variables)
- **`.env.example`** - Configuration template (updated)

### 2. Documentation
- **`config/README.md`** - Module usage guide
- **`config/MIGRATION_GUIDE.md`** - Complete migration guide for existing code
- **`config/QUICK_REFERENCE.md`** - Quick reference for developers

### 3. Utilities
- **`config/examples.py`** - Practical usage examples
- **`config/find_hardcoded.py`** - Script to detect hardcoded values

## 📋 Environment Variables Added

### Ollama
- `OLLAMA_REMOTE_A5000` - URL serveur A5000
- `OLLAMA_REMOTE_A6000` - URL serveur A6000
- `OLLAMA_LOCAL` - URL locale
- `OLLAMA_MODEL_DEFAULT` - gemma3:4b
- `OLLAMA_MODEL_LARGE` - gemma3:27b
- `OLLAMA_MODEL_MISTRAL` - mistral:7b

### HuggingFace
- `HF_TOKEN` - Authentication token
- `HF_MODEL_SENTENCE_TRANSFORMER` - Sentence embeddings model
- `HF_MODEL_BLIP` - Salesforce/blip-image-captioning-large
- `HF_MODEL_CLIP` - openai/clip-vit-base-patch32

### Base Paths
- `DATA_DIR` - Data directory
- `VECTORDB_DIR` - VectorDBs directory
- `REPORT_DIR` - Reports directory
- `CONFIG_DIR` - Configuration directory

### Dataset Paths
- `COCO_DATA_DIR`, `COCO_IMAGES_DIR`
- `FLICKR_DATA_DIR`, `FLICKR_IMAGES_DIR`
- `VIZWIZ_DATA_DIR`, `VIZWIZ_IMAGES_DIR`

### Standard Filenames
- `METADATA_FILENAME` - metadata.json
- `SELECTED_IMAGES_FILENAME` - selected_1000.json
- `PERFORMANCE_FILENAME` - performance.json
- `CLUSTER_POSITIONS_FILENAME` - cluster_positions.json
- `CORPUS_FILENAME` - corpus.json

### Processing Parameters
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `MAX_TOKENS`, `MAX_NEW_TOKENS`

### BLIP Parameters
- `BLIP_NUM_BEAMS`, `BLIP_TOP_K`, `BLIP_TOP_P`
- `BLIP_TEMPERATURE`, `BLIP_REPETITION_PENALTY`
- `BLIP_MIN_LENGTH`, `BLIP_MAX_LENGTH`

### FAISS IVF Settings
- `FAISS_NPROBE` - Number of clusters to probe
- `IVF_CLUSTER_CALCULATION` - Calculation method ('sqrt', 'fixed')
- `IVF_ALPHA` - Alpha for weighting (0 < alpha < 1)

### Retrieval Parameters
- `DEFAULT_K` - Default number of results (100)
- `DEFAULT_TOP_K` - Top-k results to return (10)
- `DEFAULT_THRESHOLD` - Similarity threshold (0.0 to 1.0)
- `MAX_RETRY_ATTEMPTS` - Max retry attempts for API calls (3)

### Rocchio PRF Parameters
- `ROCCHIO_ALPHA` - Original query weight (1.0)
- `ROCCHIO_BETA` - Relevant documents weight (0.75)
- `ROCCHIO_GAMMA` - Non-relevant documents weight (0.0)

### Fusion Weights (Two Embeddings)
- `WEIGHT_CLASSIC` - Classic embedding weight (0.5)
- `WEIGHT_BLIP` - BLIP embedding weight (0.5)
- `WEIGHT_AVERAGE` - Average weight (0.5)
- `WEIGHT_BM25` - BM25 weight (0.5)

## 🔧 Config Class Features

### Utility Methods
```python
Config.get_dataset_dir(dataset_name)          # Dataset directory
Config.get_images_dir(dataset_name)           # Dataset images directory
Config.get_metadata_path(dataset_name)        # Dataset metadata path
Config.get_selected_images_path(dataset_name) # selected_1000.json path
Config.get_vectordb_index_path(db_name)       # VectorDB index path
Config.get_vectordb_metadata_path(db_name)    # VectorDB metadata path
Config.get_ollama_url(server)                 # Ollama URL ('local', 'a5000', 'a6000')
Config.get_ollama_model(model_type)           # LLM model ('default', 'large', 'mistral')
Config.get_blip_generation_params()           # All BLIP params as dict
Config.get_rocchio_params()                   # Rocchio params (alpha, beta, gamma)
Config.get_fusion_weights()                   # Embedding fusion weights
```

### Direct Attributes (Path objects)
```python
Config.PROJECT_ROOT      # Project root
Config.DATA_DIR          # data/
Config.VECTORDB_DIR      # VectorDBs/
Config.COCO_DATA_DIR     # data/COCO/
Config.FLICKR_DATA_DIR   # data/Flickr/
# ... etc
```

### Direct Attributes (strings)
```python
Config.OLLAMA_LOCAL              # http://localhost:11434/api/generate
Config.OLLAMA_MODEL_DEFAULT      # gemma3:4b
Config.HF_MODEL_BLIP             # Salesforce/blip-image-captioning-large
Config.METADATA_FILENAME         # metadata.json
# ... etc
```

### Attributs directs (numbers)
```python
Config.MAX_TOKENS               # 2048
Config.BLIP_NUM_BEAMS          # 5
Config.BLIP_TEMPERATURE        # 0.8
# ... etc
```

## 🎯 Identified Hardcoded Values (to migrate)

### High Priority (URLs and credentials)
1. `src/retrieval/retriever_vote.py` - URLs Ollama, modèle BLIP
2. `src/retrieval/retriever_prf.py` - URL Ollama, modèle BLIP
3. `src/retrieval/retriever_structured_unstructured.py` - URLs, modèles
4. `src/methods_indexer/indexer_ivf_llm_cc.py` - URLs, modèles, chemins

### Medium Priority (paths)
5. `src/performance/performance_ivf_llm_desc.py`
6. `src/performance/performance_two_embeddings.py`
7. `src/performance/performance_vote.py`
8. `src/performance/performance_structured_unstructured.py`
9. `src/performance/performance_prf.py`

### Low Priority (parameters)
10. `src/methods_indexer/flickr_blip_desc.py`
11. `src/retrieval/retriever_two_embeddings.py`
12. `src/retrieval/retriever.py`

## 📖 How to Use

### 1. Installation
```bash
# Ensure python-dotenv is installed
pip install python-dotenv
```

### 2. Configuration
```bash
# Copy the template
cp .env.example .env

# Edit .env with your values
nano .env  # or your preferred editor
```

### 3. Import in your code
```python
from config.config import Config

# Use utility methods
url = Config.get_ollama_url('a6000')
model = Config.get_ollama_model('large')
data_dir = Config.get_dataset_dir('COCO')

# Or direct attributes
blip_model = Config.HF_MODEL_BLIP
max_tokens = Config.MAX_TOKENS
```

### 4. Test
```bash
# Display configuration
python config/config.py

# See examples
python config/examples.py

# Find remaining hardcoded values
python config/find_hardcoded.py
```

## 🔍 Migration Tools

### Detection Script
```bash
python config/find_hardcoded.py
```
Generates:
- Console report with statistics
- `config/hardcoded_values_report.md` file with details
- Prioritized task list

### Migration Guides
- `config/MIGRATION_GUIDE.md` - Guide complet avec exemples
- `config/QUICK_REFERENCE.md` - Référence rapide

## ✨ Advantages

1. **Centralization** - All parameters in one place
2. **Flexibility** - Easy environment switching (local ↔ remote)
3. **Security** - Credentials in .env (not versioned via .gitignore)
4. **Maintenance** - No more searching for hardcoded values
5. **Documentation** - Clear centralized default values
6. **Type-safe** - Automatic type conversion (int, float, Path)
7. **Reusable** - Utility methods for common patterns
8. **Scalable** - Easy to add new variables

## 🚀 Next Steps

1. **Verify configuration**
   ```bash
   python config/config.py
   ```

2. **Identify code to migrate**
   ```bash
   python config/find_hardcoded.py
   ```

3. **Migrate priority files** (see list above)
   - Start with URLs and credentials (high priority)
   - Then data paths (medium priority)
   - Finally parameters (low priority)

4. **Test after migration**
   - Verify all tests pass
   - Ensure paths are correct
   - Validate server connections

5. **Documentation**
   - Update main README.md if necessary
   - Document changes in CHANGELOG

## 📝 Important Notes

- **`.env` is NOT versioned** (in .gitignore)
- **`.env.example` IS versioned** (template for the team)
- Default values in `config.py` are fallbacks
- Always use utility methods when available
- Paths are `Path` objects (pathlib), not strings

## 🆘 Support

- Usage questions: see `config/README.md`
- Migration guide: see `config/MIGRATION_GUIDE.md`
- Quick reference: see `config/QUICK_REFERENCE.md`
- Code examples: see `config/examples.py`
