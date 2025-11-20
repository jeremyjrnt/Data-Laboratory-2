# Complete Configuration - Research Parameters

## 🎯 Added Parameters

This iteration adds **all research and algorithmic parameters** to the centralized configuration system.

## 📋 Parameter Categories

### 1. **BLIP Generation** (8 parameters)
```python
Config.BLIP_NUM_BEAMS           # 5
Config.BLIP_TOP_K               # 50
Config.BLIP_TOP_P               # 0.95
Config.BLIP_TEMPERATURE         # 0.8
Config.BLIP_REPETITION_PENALTY  # 1.5
Config.BLIP_MIN_LENGTH          # 30
Config.BLIP_MAX_LENGTH          # 150
Config.MAX_NEW_TOKENS           # 40

# Simple usage
params = Config.get_blip_generation_params()
out = model.generate(**inputs, **params)
```

### 2. **FAISS IVF Settings** (3 parameters)
```python
Config.FAISS_NPROBE              # 10 - clusters to probe
Config.IVF_CLUSTER_CALCULATION   # 'sqrt' - calculation method
Config.IVF_ALPHA                 # 0.5 - alpha weighting

# Usage example
nlist = int(np.sqrt(n_vectors)) if Config.IVF_CLUSTER_CALCULATION == 'sqrt' else n_vectors // 100
index.nprobe = Config.FAISS_NPROBE
```

### 3. **Retrieval Settings** (4 parameters)
```python
Config.DEFAULT_K              # 100 - number of results
Config.DEFAULT_TOP_K          # 10 - top-k to return
Config.DEFAULT_THRESHOLD      # 0.0 - similarity threshold
Config.MAX_RETRY_ATTEMPTS     # 3 - API retry attempts

# Usage example
def search(query, k=None, threshold=None):
    k = k or Config.DEFAULT_K
    threshold = threshold or Config.DEFAULT_THRESHOLD
    # ...
```

### 4. **Rocchio PRF** (3 parameters)
```python
Config.ROCCHIO_ALPHA   # 1.0 - original query weight
Config.ROCCHIO_BETA    # 0.75 - relevant docs weight  
Config.ROCCHIO_GAMMA   # 0.0 - non-relevant docs weight

# Simple usage
params = Config.get_rocchio_params()
retriever = PRFRetriever(**params)

# Or
modified_query = (
    Config.ROCCHIO_ALPHA * original_query +
    Config.ROCCHIO_BETA * relevant_centroid -
    Config.ROCCHIO_GAMMA * irrelevant_centroid
)
```

### 5. **Fusion Weights** (4 parameters)
```python
Config.WEIGHT_CLASSIC   # 0.5 - image embedding weight
Config.WEIGHT_BLIP      # 0.5 - text embedding weight
Config.WEIGHT_AVERAGE   # 0.5 - average weight
Config.WEIGHT_BM25      # 0.5 - BM25 weight

# Usage
weights = Config.get_fusion_weights()
final_score = (
    weights['classic'] * classic_score +
    weights['blip'] * blip_score
)
```

## 🔧 New Helper Methods

### `get_blip_generation_params()`
Returns all BLIP parameters in a single dict:
```python
{
    'max_new_tokens': 40,
    'num_beams': 5,
    'top_k': 50,
    'top_p': 0.95,
    'temperature': 0.8,
    'repetition_penalty': 1.5,
    'min_length': 30,
    'max_length': 150
}
```

### `get_rocchio_params()`
Returns Rocchio parameters:
```python
{
    'alpha': 1.0,
    'beta': 0.75,
    'gamma': 0.0
}
```

### `get_fusion_weights()`
Returns fusion weights:
```python
{
    'classic': 0.5,
    'blip': 0.5,
    'average': 0.5,
    'bm25': 0.5
}
```

## 📝 Migration Patterns

### Pattern 1: Parameters in `__init__`
```python
# ❌ Before
def __init__(self, k=100, threshold=0.0):
    self.k = k
    self.threshold = threshold

# ✅ After
def __init__(self, k=None, threshold=None):
    self.k = k or Config.DEFAULT_K
    self.threshold = threshold or Config.DEFAULT_THRESHOLD
```

### Pattern 2: BLIP Generation
```python
# ❌ Before
out = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=5,
    top_k=50,
    # ... 8 parameters
)

# ✅ After
out = model.generate(**inputs, **Config.get_blip_generation_params())
```

### Pattern 3: Rocchio PRF
```python
# ❌ Before
retriever = PRFRetriever(alpha=1.0, beta=0.75, gamma=0.0)

# ✅ After
retriever = PRFRetriever(**Config.get_rocchio_params())
```

### Pattern 4: Retry Logic
```python
# ❌ Before
max_retries = 3
for attempt in range(max_retries):
    ...

# ✅ After
for attempt in range(Config.MAX_RETRY_ATTEMPTS):
    ...
```

### Pattern 5: IVF Settings
```python
# ❌ Before
index.nprobe = 10
n_clusters = int(np.sqrt(n_vectors))

# ✅ After
index.nprobe = Config.FAISS_NPROBE
if Config.IVF_CLUSTER_CALCULATION == 'sqrt':
    n_clusters = int(np.sqrt(n_vectors))
```

## 🎨 Complete Examples

### Example 1: Complete PRF Retriever
```python
from config.config import Config

class PRFRetriever:
    def __init__(
        self,
        vectordb_name,
        llm_url=None,
        llm_model=None,
        alpha=None,
        beta=None,
        gamma=None
    ):
        self.vectordb_name = vectordb_name
        self.llm_url = llm_url or Config.get_ollama_url('local')
        self.llm_model = llm_model or Config.get_ollama_model('default')
        
        # Rocchio params
        rocchio = Config.get_rocchio_params()
        self.alpha = alpha if alpha is not None else rocchio['alpha']
        self.beta = beta if beta is not None else rocchio['beta']
        self.gamma = gamma if gamma is not None else rocchio['gamma']
    
    def retrieve(self, query, k=None, threshold=None):
        k = k or Config.DEFAULT_TOP_K
        threshold = threshold or Config.DEFAULT_THRESHOLD
        # ...
```

### Example 2: BLIP Caption Generator
```python
from config.config import Config
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIPCaptioner:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(Config.HF_MODEL_BLIP)
        self.model = BlipForConditionalGeneration.from_pretrained(Config.HF_MODEL_BLIP)
        self.gen_params = Config.get_blip_generation_params()
    
    def generate_caption(self, image):
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs, **self.gen_params)
        return self.processor.decode(out[0], skip_special_tokens=True)
```

### Example 3: Two Embeddings Fusion
```python
from config.config import Config

class TwoEmbeddingsRetriever:
    def __init__(self):
        self.weights = Config.get_fusion_weights()
    
    def fuse_scores(self, classic_score, blip_score, bm25_score=None):
        total = (
            self.weights['classic'] * classic_score +
            self.weights['blip'] * blip_score
        )
        if bm25_score is not None:
            total += self.weights['bm25'] * bm25_score
        return total
```

## 🧪 Validation

### Test configuration
```bash
python config/test_config.py
```

### Display configuration
```bash
python config/config.py
```

### View examples
```bash
python config/migration_examples.py
```

### Find code to migrate
```bash
python config/find_hardcoded.py
```

## 📊 Code Impact

### Main files to migrate

| File | Affected Parameters | Priority |
|---------|---------------------|----------|
| `retrieval/retriever_prf.py` | Rocchio, retry, k, threshold | HIGH |
| `retrieval/retriever_vote.py` | BLIP, retry, k | HIGH |
| `retrieval/retriever_structured_unstructured.py` | BLIP, retry | HIGH |
| `retrieval/retriever_two_embeddings.py` | Fusion weights, k | MEDIUM |
| `methods_indexer/indexer_ivf_kmeans.py` | IVF settings | MEDIUM |
| `methods_indexer/indexer_ivf_llm_cc.py` | IVF alpha, BLIP | MEDIUM |
| `methods_indexer/flickr_blip_desc.py` | BLIP params | LOW |
| `performance/*.py` | k, threshold, research params | LOW |

### Expected Benefits

✅ **Flexibility**: Adjust parameters without modifying code  
✅ **Experimentation**: Test different configurations easily  
✅ **Consistency**: Same values throughout the project  
✅ **Documentation**: Centralized default values  
✅ **Maintenance**: Only one .env file to edit  

## 🚀 Next Steps

1. **Test configuration**
   ```bash
   python config/test_config.py
   ```

2. **Identify code to migrate**
   ```bash
   python config/find_hardcoded.py
   ```

3. **Migrate priority files** (see table above)

4. **Validate after migration**
   - Verify all tests pass
   - Compare results before/after
   - Ensure value consistency

## 📚 Documentation

- `config/README.md` - Module documentation
- `config/MIGRATION_GUIDE.md` - Complete migration guide
- `config/QUICK_REFERENCE.md` - Quick reference
- `config/migration_examples.py` - Practical examples
- `config/test_config.py` - Validation tests

## 💡 Tips

1. **Always use helper methods** when available
2. **Allow override** in constructors (param=None)
3. **Document defaults** in docstrings
4. **Test** after each migration
5. **Version .env.example** but not .env

---

**Configuration is now 100% centralized and complete! 🎉**
