# Evaluating LLM and VLM Integration Strategies for Text-to-Image Retrieval

A comprehensive research project investigating how Large Language Models (LLMs) and Vision-Language Models (VLMs) can be integrated into text-to-image retrieval systems to overcome the inherent limitations of textual queries in shared embedding spaces.

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Performance Analysis](#performance-analysis)
- [Dependencies](#dependencies)
- [Citation](#citation)

## Overview

This project explores the integration of LLMs and VLMs into text-to-image retrieval pipelines to address the semantic gap between textual queries and visual content. While traditional systems like CLIP provide a shared embedding space for images and text, human-written captions are often short, generic, and under-specified, leading to ambiguous matches and suboptimal retrieval performance.

### Abstract

In text-to-image retrieval with a shared image–text embedding space (as in CLIP), using a textual description as the query comes with intrinsic limitations: human-written captions are short, generic, and under-specified, whereas image embeddings encode rich visual content. As a result, many images may appear compatible with the same caption, and ambiguous terms can align with undesired visual patterns. This motivates exploring whether Large Language Models (LLMs) and vision–language models (VLMs) can compensate for the semantic limitations of textual queries.

We design and evaluate **four methods** integrating LLMs/VLMs at different stages of the retrieval pipeline:

**Late-Stage Methods** (Post-Retrieval):
1. **LLM as Relevance Feedback Provider**: Uses LLM to judge relevance of top-k results, then applies Rocchio algorithm to update query embedding
2. **LLM as Query Reformulator**: LLM reformulates queries based on top-k results in structured (scene/emotion/action) or unstructured paradigms

**Early-Stage Methods** (Index-Time Integration):
3. **BLIP-Based Multiple Retrieval Systems**: Combines 4 retrieval channels (Image embeddings, BLIP captions, averaged embeddings, BM25) via fusion methods (CombSUM, RRF, Borda)
4. **Inverted File Index with BLIP-Based Cluster Corpus**: Uses FPS sampling + LLM aggregation to create semantic cluster descriptions for IVF routing

**Key Results:**
- **Late-stage methods** improve high-rank metrics (R@5, R@10, MRR) but remain **unstable** with >50% queries experiencing rank degradations
- **Early-stage hybrid retrieval** yields **strong and stable gains** on moderate-scale datasets: R@10 improves from 0.112 to **0.330** (+195%) and MRR from 0.052 to **0.149** (+186%) on Flickr30k
- **Scaling to large IVF indexes** proves challenging and leads to severe performance regressions across all metrics
- **Larger LLMs** (Gemma3 27B) consistently outperform smaller ones, though gains are not strictly proportional to model size
- **Improvements are most pronounced on VizWiz** (noisiest dataset), suggesting LLM/VLM support is especially valuable when visual signals are weak
- **Trade-off**: All methods degrade mean rank while improving high-rank metrics, indicating that gains at top positions come at the cost of extreme degradations for a subset of queries

## Research Motivation

Many applications rely on retrieving images from textual descriptions, including:
- Personal photo search
- Media management systems
- Recommendation engines
- Content discovery platforms

Users typically expect their desired image to appear among the top ranks of the returned list. Most modern systems are built on multimodal embedding spaces (e.g., CLIP) that provide shared representations for images and text. However, these systems face fundamental challenges:

1. **Semantic Gap**: Human-written captions are inherently brief and generic, while images contain rich visual details
2. **Ambiguity**: Multiple images may match the same textual description
3. **Under-specification**: Queries often lack sufficient detail to uniquely identify the desired image
4. **Visual Pattern Misalignment**: Ambiguous terms can align with unintended visual patterns

Recent VLMs and LLMs exhibit impressive semantic understanding and hold strong potential for text-to-image retrieval, but it remains unclear:
- **How** they should be integrated into the retrieval pipeline
- **How much** they can help in realistic, computationally tractable settings
- **When** their benefits are most pronounced

This work provides a unified framework for understanding these questions through comprehensive empirical analysis.

## Project Structure

```
DataLab2Project/
├── data/                           # Dataset files and metadata
│   ├── COCO/                       # MS COCO dataset
│   │   ├── images/                 # Image files
│   │   ├── coco_metadata.json      # Processed metadata with captions
│   │   ├── cluster_positions.json  # IVF cluster assignments
│   │   ├── COCO_script.py          # Dataset processing script
│   │   └── selected_1000.json      # Evaluation subset
│   ├── Flickr/                     # Flickr30K dataset
│   │   ├── images/                 # Image files
│   │   ├── flickr_metadata.json    # Processed metadata
│   │   ├── Flickr_script.py        # Dataset processing script
│   │   └── selected_1000.json      # Evaluation subset
│   └── VizWiz/                     # VizWiz dataset
│       ├── images/                 # Image files
│       ├── VizWiz_metadata.json    # Processed metadata
│       ├── VizWiz_script.py        # Dataset processing script
│       └── selected_1000.json      # Evaluation subset
│
├── src/                            # Source code
│   ├── helpers/                    # Utility functions and shared components
│   ├── indexer/                    # Vector index construction
│   │   ├── indexer_two_embeddings.py      # Build hybrid BLIP+Caption indexes
│   │   ├── indexer_ivf_kmeans.py          # Build IVF k-means clustered indexes
│   │   ├── indexer_ivf_llm_cc.py          # Build IVF with LLM cluster summaries
│   │   └── flickr_blip_desc.py            # Generate BLIP captions for Flickr
│   ├── performance/                # Evaluation metrics and benchmarking
│   │   ├── performance.py                 # General retrieval metrics (R@K, MRR)
│   │   └── performance_ivf_llm_desc.py    # IVF-specific evaluation
│   ├── retrieval/                  # Retrieval method implementations
│   │   ├── retriever.py                   # Base retriever class
│   │   ├── retriever_prf.py               # Pseudo-Relevance Feedback
│   │   ├── retriever_structured.py        # Structured query reformulation
│   │   ├── retriever_unstructured.py      # Unstructured query reformulation
│   │   ├── retriever_structured_unstructured.py  # Combined reformulation
│   │   ├── retriever_two_embeddings.py    # Hybrid double-embedding retrieval
│   │   ├── retriever_ivf_llm_desc.py      # IVF with LLM routing
│   │   └── retriever_vote.py              # Voting ensemble method
│   ├── ui/                         # User interface components
│   └── vector_store/               # Vector database management
│       ├── indexer.py                     # FAISS index builder (CPU)
│       ├── gpu_indexer.py                 # FAISS index builder (GPU)
│       └── inspector.py                   # VectorDB inspection utilities
│
├── VectorDBs/                      # Pre-built vector indexes
│   ├── COCO_VectorDB.index         # COCO FAISS index
│   ├── COCO_IVF_KMeans.index       # COCO IVF index
│   ├── Flickr_VectorDB.index       # Flickr FAISS index
│   ├── Flickr_Double_Embeddings.index  # Hybrid retrieval index
│   └── VizWiz_VectorDB.index       # VizWiz FAISS index
│
├── report/                         # Performance analysis
│   ├── performance_raw/            # Raw experimental results
│   ├── performance_prf/            # Pseudo-relevance feedback results
│   ├── performance_ivf_llm_desc/   # IVF + LLM description results
│   ├── performance_double_embeddings/  # Hybrid retrieval results
│   ├── performance_structured_unstructured/  # Query reformulation results
│   └── performance_vote/           # Voting ensemble results
│
├── visualizations/                 # Visualization scripts and outputs
│   ├── create_sample_grids.py      # Generate dataset sample grids
│   ├── ivf_llm_cc/                 # IVF method visualizations
│   ├── prf/                        # PRF method visualizations
│   ├── two_embeddings/             # Hybrid method visualizations
│   ├── un_structured/              # Query reformulation visualizations
│   └── vote/                       # Voting method visualizations
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Datasets

This project evaluates retrieval methods across three diverse datasets, each presenting unique challenges:

### 1. MS COCO 2017 (Common Objects in Context)
- **Size**: ~118K images with 5 captions each (only first caption used for retrieval)
- **Characteristics**: Moderately challenging everyday images with broad coverage of objects, contexts, and visual compositions
- **Captions**: Short, natural, human-written descriptions
- **Difficulty**: Baseline retrieval performance serves as reference for general-purpose multimodal retrieval
- **Observation**: Only Gemma3 27B consistently produces clear gains on high-rank metrics; smaller LLMs show inconsistent improvements

### 2. Flickr30K
- **Size**: ~31K images with 5 captions each (only first caption used for retrieval)
- **Characteristics**: People-centric everyday activities and events with complex object-action relationships
- **Captions**: Short, natural, crowd-sourced descriptions
- **Difficulty**: Relative homogeneity increases ambiguity across images with similar captions
- **Observation**: Best dataset for hybrid early-stage methods; achieves largest absolute gains (R@10: 0.112 → 0.330)

### 3. VizWiz
- **Size**: ~30K images from blind users (corrupted images filtered)
- **Characteristics**: Real-world photos with deliberate imperfections (blur, low light, off-center framing, occlusions)
- **Captions**: Short, crowd-sourced questions/descriptions
- **Difficulty**: Most challenging dataset due to noisy, imperfect images and weak visual signals
- **Observation**: Largest **relative** improvements from LLM/VLM integration; late-stage methods most effective here, suggesting these approaches excel when visual quality is poor

### Dataset Preprocessing

Each dataset includes:
- **Metadata JSON**: Processed metadata with image IDs, filenames, captions, and baseline rankings
- **Selected subsets**: 1000-image evaluation sets for consistent benchmarking
- **Corruption filtering**: Removal of corrupted or invalid images (especially VizWiz)
- **Caption selection**: Systematic selection of representative captions from multiple options

## Methodology

We evaluate **four distinct integration strategies** for LLMs and VLMs in text-to-image retrieval:

### Late-Stage Methods (Post-Retrieval)

#### 1. LLM as Relevance Feedback Provider (Rocchio Algorithm)
- **Pipeline**: 
  1. Retrieve top-10 images using baseline CLIP
  2. Generate BLIP captions for retrieved images
  3. LLM judges which images are relevant based on semantic alignment with query
  4. Apply Rocchio algorithm to update query embedding (shift toward relevant centroids)
  5. Perform second retrieval with updated query vector
- **Rationale**: Automate relevance feedback without human interaction; leverage LLM semantic understanding to identify relevant samples
- **Implementation**: Tested with Gemma3 4B, Mistral 7B, Gemma3 27B
- **Results**:
  - ✓ Systematic boost on early-rank metrics (R@5, R@10, MRR) except COCO
  - ✗ No-degradation rate always <0.5 (degrades >50% of queries)
  - ✗ Mean rank substantially worse than baseline (mean vs median divergence)
  - ✓ VizWiz shows best improvements (MRR: 0.018 → 0.070 with Gemma3 27B)
  - Pattern: Gemma3 27B > Mistral 7B > Gemma3 4B across datasets

#### 2. LLM as Query Reformulator
- **Pipeline**:
  1. Retrieve top-3 images using baseline CLIP (k=3 to reduce noise)
  2. Generate BLIP captions for top-3 results
  3. LLM reformulates query based on original query + top-3 captions
  4. Perform second retrieval with reformulated query
- **Variants**:
  - **Structured**: LLM extracts 3 semantic features (scene, emotion, action) from query and top-k captions, then reformulates by identifying missing elements and avoiding drift
  - **Unstructured**: LLM receives raw BLIP captions and reformulates directly without explicit feature extraction
- **Rationale**: Fine-grained feedback at feature level; preserve key query elements (e.g., "young girl" vs "child") while enriching semantics
- **Implementation**: Tested with Gemma3 4B, Mistral 7B, Gemma3 27B
- **Results**:
  - ✓ Consistently improves high-rank metrics over baseline (except COCO with smaller LLMs)
  - ✓ Structured paradigm slightly outperforms unstructured on COCO and Flickr
  - ✓ VizWiz shows largest relative gains; both paradigms perform similarly (noisy images reduce structured advantage)
  - ✓ Gemma3 4B achieves strong results on COCO (comparable to 27B) but 27B dominates on VizWiz
  - ✗ Mean rank always worse; long negative tail of extreme degradations
  - ✓ Median progression consistently positive on VizWiz
  - Best configuration: Gemma3 27B structured on VizWiz (R@10: 0.051 → 0.126, MRR: 0.018 → 0.074)

### Early-Stage Methods (Index-Time Integration)

#### 3. BLIP-Based Multiple Retrieval Systems (Hybrid)
- **Scope**: Moderate-scale datasets only (<50K images) due to BLIP inference cost
- **Evaluated on**: Flickr30k (~31K images)
- **Index Construction**:
  1. Generate BLIP captions for all images (single caption + 5-caption corpus)
  2. Create 4 separate retrieval systems:
     - **Image**: CLIP embeddings of images
     - **BLIP**: CLIP embeddings of BLIP-generated captions
     - **Average**: Normalized arithmetic mean of image + BLIP embeddings
     - **BM25**: Textual corpus of 5 diverse BLIP captions per image
  3. Build FAISS indexes for embedding-based systems
- **Fusion Methods**: CombSUM, RRF (Reciprocal Rank Fusion), Borda
- **Rationale**: Exploit both visual and textual modalities; modern hybrid search combines multiple signals
- **Results**:
  - ✓ **Best overall method**: Substantial and stable improvements
  - ✓ Image+BLIP+BM25 with Borda: R@10 = **0.330** (+195%), MRR = 0.144
  - ✓ Image+BLIP+BM25 with RRF: MRR = **0.149** (+186%), Median Rank = **25** (vs 49.5 baseline)
  - ✓ Average system alone: R@1 = **0.078**, MRR = **0.149** (best single system)
  - ✓ Improvement rate up to **0.583** (Image+BLIP+BM25 Borda)
  - ✓ Several configurations avoid long negative tail (Image+BLIP RRF, Image+BLIP+BM25 RRF)
  - ✓ Mean LRR consistently positive for best configurations (+0.679 for Image+BLIP+BM25 RRF)
  - ⚠️ Some configurations still exhibit instability (mean-median divergence)
  - **Recommendation**: Image+BLIP+BM25 with RRF offers best balance of performance and stability

#### 4. Inverted File Index with BLIP-Based Cluster Corpus
- **Scope**: Large-scale datasets (>50K images) where exhaustive BLIP captioning is impractical
- **Evaluated on**: MS COCO (~118K images)
- **Objective**: Improve cluster ranking in IVF-Flat index by adding text-based retrieval signals
- **Index Construction**:
  1. Apply k-means clustering to create IVF index
  2. For each cluster, use **Farthest Point Sampling (FPS)** to select representative images:
     - Initialize from medoid (semantic anchor)
     - Iteratively select farthest point from all previously chosen points
     - Controlled by overall sampling budget to limit BLIP inferences
  3. Generate BLIP captions for sampled points only
  4. LLM aggregates these captions into cluster semantic description
  5. Build BM25 index over cluster documents
- **Retrieval**: Rank cluster centroids using both embedding similarity + BM25 scores, fused via CombSUM/RRF/Borda
- **Rationale**: FPS captures semantic richness and diversity within clusters; avoid captioning every image
- **Implementation**: Tested with Gemma3 4B, Mistral 7B, Gemma3 27B
- **Results**:
  - ✗ **Severe performance regression** across all metrics and all LLMs
  - ✗ Baseline R@1 = 0.129 vs best method = 0.040 (Gemma3 27B RRF)
  - ✗ Baseline R@10 = 0.601 vs best method = 0.255 (Gemma3 27B RRF)
  - ✗ Mean/Median ranks more than doubled compared to baseline
  - ✗ Improvement rate always <0.20; No-degradation rate <0.21
  - ✗ Mean LRR strongly negative (worst: -2.359 for Mistral 7B Borda)
  - **Conclusion**: LLM-based hybrid fusion for IVF cluster ranking is detrimental
  - **Hypothesized causes**: Cluster heterogeneity, textual corpus too small/diluted, FPS sampling may not capture fine-grained semantics, LLM aggregation loses critical details
  - **Status**: Requires fundamental redesign; current implementation not viable

### Voting Ensemble
- **Pipeline**: Combine multiple methods through majority voting
- **Rationale**: Aggregate strengths of different approaches
- **Implementation**: Vote across top-k results from different methods

## Key Findings

### Performance Summary

**Best Results by Method:**

| Method | Dataset | Config | R@1 | R@5 | R@10 | MRR | Improv. Rate | Mean LRR |
|--------|---------|--------|-----|-----|------|-----|--------------|----------|
| **Baseline** | Flickr | - | 0.000 | 0.069 | 0.112 | 0.052 | - | - |
| **Baseline** | COCO | - | 0.000 | 0.022 | 0.037 | 0.014 | - | - |
| **Baseline** | VizWiz | - | 0.000 | 0.033 | 0.051 | 0.018 | - | - |
| | | | | | | | | |
| **Rocchio (Late)** | VizWiz | Gemma3 27B | 0.041 | 0.089 | 0.117 | **0.070** | 0.485 | 0.229 |
| **Rocchio (Late)** | Flickr | Gemma3 27B | 0.036 | 0.106 | 0.175 | **0.087** | 0.459 | -0.375 |
| | | | | | | | | |
| **Query Reform. (Late)** | VizWiz | Gemma3 27B Struct. | 0.044 | **0.095** | **0.126** | **0.074** | **0.571** | **0.807** |
| **Query Reform. (Late)** | Flickr | Mistral 7B Struct. | 0.032 | **0.126** | **0.202** | **0.094** | **0.504** | -0.129 |
| **Query Reform. (Late)** | COCO | Gemma3 27B Struct. | 0.007 | 0.031 | 0.040 | 0.022 | **0.510** | **0.186** |
| | | | | | | | | |
| **Hybrid Multi-System (Early)** | Flickr | Img+BLIP+BM25 Borda | 0.053 | 0.212 | **0.330** | 0.144 | **0.583** | 0.275 |
| **Hybrid Multi-System (Early)** | Flickr | Img+BLIP+BM25 RRF | 0.063 | **0.225** | 0.326 | **0.149** | 0.559 | **0.679** |
| **Hybrid Multi-System (Early)** | Flickr | Average only | **0.078** | 0.205 | 0.289 | **0.149** | 0.527 | 0.064 |
| | | | | | | | | |
| **IVF+LLM (Early)** | COCO | *All configs fail* | <0.040 | <0.255 | <0.471 | <0.066 | <0.183 | <-1.33 |
| **IVF+LLM Baseline** | COCO | - | **0.129** | **0.601** | **0.776** | **0.208** | - | - |

**Key Observations:**
- 🏆 **Best absolute gains**: Hybrid Multi-System on Flickr (R@10: +195%, MRR: +186%)
- 🏆 **Most stable**: Hybrid Multi-System with RRF fusion (positive Mean LRR, minimal degradations)
- 🏆 **Best late-stage**: Query Reformulation (Structured) consistently outperforms Rocchio
- ⚠️ **All methods degrade mean rank** while improving high-rank metrics
- ❌ **IVF+LLM completely fails** on COCO (requires redesign)

### Key Insights

1. **Early-Stage Hybrid Retrieval is Most Effective (on Moderate-Scale Data)**
   - BLIP-based multi-system approach shows strongest and most stable improvements
   - Best configuration: Image+BLIP+BM25 with RRF fusion
   - R@10 improvement: +195% on Flickr (0.112 → 0.330)
   - MRR improvement: +186% on Flickr (0.052 → 0.149)
   - Median rank improvement: 2× better (49.5 → 25)
   - Mean LRR consistently positive (+0.679 for best config)
   - Improvement rate up to 58.3%
   - Limited to datasets <50K images due to BLIP inference cost

2. **Late-Stage Methods: High Gains but High Instability**
   - Query Reformulation outperforms Rocchio relevance feedback
   - Structured paradigm > Unstructured (except on VizWiz)
   - Consistent improvements on R@5, R@10, MRR across datasets
   - **Critical limitation**: No-degradation rate always <0.5
   - Mean rank always degrades despite improved high-rank metrics
   - Long negative tail of extreme degradations (visible in rank progression distributions)
   - Median progression often positive, indicating method helps majority but fails catastrophically on subset

3. **Scaling to Large IVF Indexes Fails Completely**
   - IVF+LLM cluster method shows severe performance regression
   - All metrics worse than baseline across all LLMs and fusion methods
   - Baseline R@10 = 0.601 vs best method = 0.255 (-58% degradation)
   - Improvement rate <20%, No-degradation rate <21%
   - Mean LRR strongly negative (-2.359 worst case)
   - Root causes likely: cluster heterogeneity, diluted textual corpus, FPS sampling limitations, LLM aggregation losing fine-grained semantics
   - Current implementation not viable; requires fundamental redesign

4. **LLM Size Effect: Larger is Better (But Not Linearly)**
   - Gemma3 27B consistently outperforms smaller LLMs across all methods and datasets
   - Exception: Gemma3 4B performs comparably to 27B on COCO (easier dataset)
   - Mistral 7B (older generation) lags behind Gemma models of similar size
   - Gap most pronounced on VizWiz (hardest dataset)
   - Gains not proportional to parameter count (4B→27B is 6.75× params but <2× performance gain)
   - Diminishing returns suggest 27B may be near optimal for this task
   - Model generation/training may matter more than raw size (Gemma3 vs Mistral)

5. **Dataset Difficulty Predicts LLM/VLM Benefit**
   - **VizWiz (noisiest)**: Largest relative improvements from all methods
     - Query Reform. Structured: +147% R@10 (0.051 → 0.126)
     - Rocchio: +129% R@10 (0.051 → 0.117)
     - Mean LRR consistently positive (up to +0.807)
   - **Flickr (medium)**: Largest absolute gains on hybrid method
     - Hybrid Multi-System: +195% R@10 (0.112 → 0.330)
   - **COCO (cleanest)**: Smallest improvements; only Gemma3 27B shows consistent gains
     - Smaller LLMs struggle or degrade performance
   - **Conclusion**: LLM/VLM support most valuable when visual signals are weak or ambiguous
   - Cleaner datasets with strong CLIP baseline benefit less from LLM integration

6. **Structured vs Unstructured Reformulation**
   - Structured (scene/emotion/action extraction) outperforms unstructured on COCO and Flickr
   - No clear advantage on VizWiz (both paradigms perform similarly)
   - Hypothesis: Noisy images make feature extraction less reliable → structured approach loses advantage
   - Structured approach requires more LLM reasoning steps → may amplify errors on poor visual input
   - Best practice: Use structured on clean datasets, either approach on noisy data

7. **Universal Trade-off: Top-Rank Gains vs Mean Degradation**
   - All methods (including best configurations) degrade mean rank
   - Improvements concentrated at top positions (R@1, R@5, R@10)
   - Long tail of catastrophic failures inflates mean rank
   - Median rank often improves, indicating majority of queries benefit
   - Trade-off may be acceptable for real-world use cases where top-k matters most
   - But high variance introduces risk: some queries fail severely

8. **Fusion Method Selection Matters**
   - RRF (Reciprocal Rank Fusion) shows best stability across metrics
   - CombSUM competitive on high-rank metrics but less stable
   - Borda achieves highest R@10 but worst mean-median divergence
   - Recommendation: RRF for production (balance of performance + stability)

9. **Single-System Baselines Can Be Strong**
   - Average embedding (image + BLIP) alone achieves MRR = 0.149 (matching best fusion)
   - Simpler approach with lower computational cost
   - May be preferable when inference latency is critical
   - Fusion methods justify cost only when R@10 or improvement rate is priority

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for CLIP/BLIP inference)
- 32GB+ RAM (for large vector indexes)
- 100GB+ disk space (for datasets and indexes)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/jeremyjrnt/Data-Laboratory-2.git
cd Data-Laboratory-2
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets**
```bash
# COCO
python data/COCO/COCO_script.py

# Flickr30K
python data/Flickr/Flickr_script.py

# VizWiz
python data/VizWiz/VizWiz_script.py
```

5. **Build vector indexes** (optional, pre-built indexes available)
```bash
python src/indexer/build_indexes.py
```

## Usage

### Core Components Overview

#### Retrieval Methods (`src/retrieval/`)

The project implements multiple retrieval strategies, each in its own module:

**1. Base Retriever (`retriever.py`)**
- Foundation class for all retrieval methods
- Handles CLIP embedding generation
- Performs FAISS similarity search
- Returns ranked image results

**2. Pseudo-Relevance Feedback (`retriever_prf.py`)**
- Retrieves initial candidates using CLIP
- Extracts captions from top-k results
- Uses LLM (via Ollama) to re-rank based on semantic relevance
- Supports multiple LLM models (Gemma, Mistral, Llama)

**3. Query Reformulation**
- **Structured (`retriever_structured.py`)**: LLM generates structured attributes (objects, colors, actions, scene)
- **Unstructured (`retriever_unstructured.py`)**: LLM generates natural language query expansion
- **Combined (`retriever_structured_unstructured.py`)**: Tests both approaches

**4. Two Embeddings Hybrid (`retriever_two_embeddings.py`)**
- Queries two parallel FAISS indexes:
  - Classic index: CLIP embeddings from image captions
  - BLIP index: CLIP embeddings from VLM-generated descriptions
- Merges results with configurable weighting
- Combines complementary semantic signals

**5. IVF + LLM Routing (`retriever_ivf_llm_desc.py`)**
- Uses IVF (Inverted File) index with k-means clustering
- Routes queries to relevant clusters via LLM-generated summaries
- Designed for scalability on large datasets

**6. Voting Ensemble (`retriever_vote.py`)**
- Aggregates results from multiple retrieval methods
- Implements majority voting across top-k results
- Combines strengths of different approaches

#### Indexers (`src/indexer/`)

Index builders that preprocess datasets and create FAISS vector databases:

**1. Two Embeddings Indexer (`indexer_two_embeddings.py`)**
```python
class TwoEmbeddingsIndexer:
    """
    Creates hybrid retrieval indexes with dual embedding sources:
    - Image captions → CLIP embeddings
    - BLIP-generated descriptions → CLIP embeddings
    
    Supports:
    - BLIP caption generation for entire dataset
    - Separate FAISS indexes for each embedding type
    - Averaged embeddings combining both signals
    """
```

**2. IVF K-Means Indexer (`indexer_ivf_kmeans.py`)**
```python
"""
Builds Inverted File (IVF) index using k-means clustering:
- Clusters image embeddings for faster search
- Reduces search space from O(N) to O(k + N/k)
- Optimized for large-scale retrieval
"""
```

**3. IVF + LLM Cluster Descriptions (`indexer_ivf_llm_cc.py`)**
```python
"""
Enhanced IVF indexing with LLM-generated cluster summaries:
- Performs k-means clustering on embeddings
- Samples representative images per cluster (FPS algorithm)
- Generates BLIP captions for selected images
- Uses LLM to create cluster summary descriptions
- Enables semantic cluster routing at query time

Key functions:
- fps_from_centroid(): Farthest Point Sampling for diversity
- build_selection(): Smart caption selection from clusters
- call_ollama_llm(): LLM interaction for summary generation
"""
```

**4. BLIP Caption Generator (`flickr_blip_desc.py`)**
```python
"""
Generates VLM captions for Flickr dataset:
- Loads BLIP model for image captioning
- Processes images in batches
- Saves generated captions for indexing
"""
```

#### Vector Store (`src/vector_store/`)

Low-level FAISS index management and inspection:

**1. FAISS Indexer (`indexer.py`)**
```python
class FAISSVectorIndexer:
    """
    CPU-based FAISS index builder:
    - Creates IndexFlatL2 or IndexFlatIP indexes
    - Manages metadata association
    - Saves/loads indexes and metadata
    - Supports incremental updates
    """
```

**2. GPU Indexer (`gpu_indexer.py`)**
```python
"""
GPU-accelerated FAISS index builder:
- Leverages CUDA for faster indexing
- Supports larger-scale datasets
- Same interface as CPU indexer
"""
```

**3. VectorDB Inspector (`inspector.py`)**
```python
class VectorDBInspector:
    """
    Utilities for inspecting and debugging FAISS indexes:
    - Display index statistics (size, dimension, metric)
    - Sample nearest neighbors
    - Validate metadata consistency
    - Diagnose index issues
    """
```

### Example Usage

#### Basic Baseline Retrieval

```python
import faiss
import clip
import torch
from pathlib import Path

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load FAISS index
index_path = Path("VectorDBs/COCO_VectorDB.index")
index = faiss.read_index(str(index_path))

# Encode query
query = "a dog playing in the park"
text_token = clip.tokenize([query]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_token)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Search
query_vector = text_features.cpu().numpy()
distances, indices = index.search(query_vector, k=10)

print(f"Top 10 results for: '{query}'")
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"{rank}. Image ID: {idx}, Distance: {dist:.4f}")
```

#### Pseudo-Relevance Feedback (PRF)

```python
from src.retrieval.retriever_prf import LLMPseudoRelevanceFeedback
from pathlib import Path

# Initialize PRF retriever
retriever = LLMPseudoRelevanceFeedback(
    dataset_name='COCO',
    llm_model='gemma3:27b',  # or 'mistral:7b', 'llama3.1:8b'
    k_feedback=5,            # Number of top results to use for feedback
    device='cuda'
)

# Load vector database
retriever.load_vectordb()

# Perform retrieval with LLM re-ranking
query = "a person riding a bicycle on a street"
results = retriever.retrieve_with_llm_reranking(
    query=query,
    k_initial=100,  # Initial retrieval pool
    k_final=10      # Final re-ranked results
)

# Results include LLM similarity scores
for rank, result in enumerate(results, 1):
    print(f"{rank}. Image {result['image_id']}: {result['llm_score']:.4f}")
```

#### Hybrid Two Embeddings Retrieval

```python
from src.retrieval.retriever_two_embeddings import TwoEmbeddingsRetriever

# Initialize with dual indexes
retriever = TwoEmbeddingsRetriever(
    dataset_name='Flickr',
    alpha=0.5,      # Weight: 0=BLIP only, 0.5=equal, 1=caption only
    device='cuda'
)

# Load both indexes (caption + BLIP)
retriever.load_classic_vectordb()  # Human captions
retriever.load_blip_vectordb()     # BLIP-generated descriptions

# Query both channels
query = "children playing soccer in a park"
results = retriever.retrieve_combined(
    query=query,
    k=10,
    merge_strategy='weighted'  # or 'interleave', 'max'
)

# Visualize which index contributed each result
for rank, result in enumerate(results, 1):
    source = "Caption" if result['from_classic'] else "BLIP"
    print(f"{rank}. [{source}] Image {result['image_id']}: {result['score']:.4f}")
```

#### Query Reformulation (Structured)

```python
from src.retrieval.retriever_structured import StructuredQueryRetriever

# Initialize with LLM for query expansion
retriever = StructuredQueryRetriever(
    dataset_name='VizWiz',
    llm_model='mistral:7b',
    device='cuda'
)

# Load index
retriever.load_vectordb()

# Reformulate query into structured attributes
query = "red car"
structured = retriever.reformulate_query(query)
# Returns: {
#   'objects': ['car', 'vehicle'],
#   'colors': ['red'],
#   'actions': [],
#   'scene': ['outdoor', 'street']
# }

# Retrieve with expanded query
expanded_query = retriever.build_expanded_query(structured)
results = retriever.retrieve(expanded_query, k=10)

print(f"Original: {query}")
print(f"Expanded: {expanded_query}")
```

#### IVF + LLM Cluster Routing

```python
from src.retrieval.retriever_ivf_llm_desc import IVFLLMRetriever

# Initialize IVF retriever with LLM cluster descriptions
retriever = IVFLLMRetriever(
    dataset_name='COCO',
    llm_model='gemma3:4b',
    n_clusters=512,      # Number of k-means clusters
    n_probe=10,          # Clusters to search per query
    device='cuda'
)

# Load IVF index and cluster summaries
retriever.load_ivf_index()
retriever.load_cluster_descriptions()

# Route query to relevant clusters using LLM
query = "a cat sitting on a window"
cluster_ids = retriever.route_query_to_clusters(query, n_clusters=5)
print(f"Routing to clusters: {cluster_ids}")

# Search within selected clusters
results = retriever.retrieve_from_clusters(query, cluster_ids, k=10)
```

#### Voting Ensemble

```python
from src.retrieval.retriever_vote import GroundTruthLLMVotingRetriever

# Initialize voting ensemble
retriever = GroundTruthLLMVotingRetriever(
    dataset_name='Flickr',
    methods=['baseline', 'prf', 'two_embeddings'],
    voting_strategy='majority',  # or 'weighted', 'rank_fusion'
    device='cuda'
)

# Load all required indexes
retriever.load_all_indexes()

# Perform ensemble retrieval
query = "a group of people hiking in mountains"
results = retriever.vote_and_rank(query, k=10)

# See which methods contributed to each result
for rank, result in enumerate(results, 1):
    votes = result['vote_count']
    methods = ', '.join(result['voted_by'])
    print(f"{rank}. Image {result['image_id']}: {votes} votes from {methods}")
```

#### Building Custom Indexes

```python
from src.indexer.indexer_two_embeddings import TwoEmbeddingsIndexer

# Initialize indexer for Flickr
indexer = TwoEmbeddingsIndexer(dataset_name='Flickr')

# Load dataset metadata
indexer.load_metadata()

# Generate BLIP captions for all images (one-time process)
indexer.generate_blip_captions()

# Create separate indexes
indexer.create_blip_caption_vectordb()   # BLIP descriptions
indexer.create_average_vectordb()        # Averaged embeddings

print("Indexes created successfully!")
```

```python
from src.indexer.indexer_ivf_llm_cc import build_ivf_with_llm_summaries

# Build IVF index with LLM-generated cluster descriptions
build_ivf_with_llm_summaries(
    dataset_name='COCO',
    n_clusters=512,
    llm_model='gemma3:27b',
    samples_per_cluster=10,
    device='cuda'
)

# This creates:
# - COCO_IVF_KMeans.index (clustered FAISS index)
# - COCO_IVF_KMeans_metadata.json (cluster assignments)
# - COCO_IVF_CC_{model}.json (LLM cluster summaries)
```

### Evaluation

```python
from src.performance.performance import evaluate_retrieval_performance

# Evaluate any retriever on test set
metrics = evaluate_retrieval_performance(
    retriever=retriever,
    dataset_name='COCO',
    selected_images_path='data/COCO/selected_1000.json',
    k_values=[1, 5, 10, 20, 50]
)

# Print results
print(f"Recall@1:  {metrics['R@1']:.4f}")
print(f"Recall@5:  {metrics['R@5']:.4f}")
print(f"Recall@10: {metrics['R@10']:.4f}")
print(f"Recall@20: {metrics['R@20']:.4f}")
print(f"MRR:       {metrics['MRR']:.4f}")

# Per-query analysis
for query_id, query_metrics in metrics['per_query'].items():
    rank = query_metrics['rank']
    improved = query_metrics['rank_improvement']
    print(f"Query {query_id}: Rank {rank} (Δ{improved:+d})")
```

```python
from src.vector_store.inspector import VectorDBInspector

# Inspect any FAISS index
inspector = VectorDBInspector(
    index_path='VectorDBs/Flickr_Double_Embeddings.index',
    metadata_path='VectorDBs/Flickr_Double_Embeddings_metadata.json'
)

# Display statistics
inspector.print_stats()
# Output:
# Index type: IndexFlatIP
# Total vectors: 31,783
# Vector dimension: 512
# Metric type: Inner Product
# Memory usage: 64.8 MB

# Sample nearest neighbors for debugging
test_vector = inspector.get_vector(image_id=42)
neighbors = inspector.find_neighbors(test_vector, k=5)
inspector.display_neighbors(neighbors)
```

## Visualization

### Generate Dataset Sample Grids

Visualize random samples from each dataset with their captions:

```bash
cd visualizations
python create_sample_grids.py
```

This generates 3x3 grids for each dataset showing:
- Random image samples
- Ground-truth captions
- Visual diversity of the dataset

Output files:
- `COCO_sample_grid.png`
- `Flickr_sample_grid.png`
- `VizWiz_sample_grid.png`

### Performance Visualizations

Each experiment folder in `report/` contains visualization scripts:

```bash
# PRF performance plots
cd report/performance_prf
python results.py

# Double embeddings analysis
cd report/performance_double_embeddings
python results.py

# IVF method analysis
cd report/performance_ivf_llm_desc
python results.py
```

Generated visualizations include:
- Recall@K curves
- MRR comparisons
- Per-query performance distributions
- Method comparison heatmaps
- Rank improvement/degradation analysis

## Performance Analysis

### Metrics

All experiments are evaluated using standard information retrieval metrics:

- **Recall@K (R@K)**: Proportion of queries where the correct image appears in top K results
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank across all queries
- **Per-query analysis**: Distribution of rank improvements and degradations

### Experimental Results

Detailed performance results are organized by method:

#### 1. Baseline Performance
- Location: `report/performance_raw/`
- CLIP-based retrieval on all three datasets
- Establishes performance ceiling for improvements

#### 2. Pseudo-Relevance Feedback
- Location: `report/performance_prf/`
- Multiple LLMs tested (Gemma 4B/27B, Mistral 7B)
- Analysis of feedback size (k=3, 5, 10)
- Per-query stability analysis

#### 3. Double Embeddings (Hybrid)
- Location: `report/performance_double_embeddings/`
- Caption vs BLIP weighting experiments
- Ablation studies on combination strategies
- Scaling analysis

#### 4. IVF + LLM Descriptions
- Location: `report/performance_ivf_llm_desc/`
- Cluster size experiments
- LLM-generated cluster summaries
- Routing accuracy analysis

#### 5. Query Reformulation
- Location: `report/performance_structured_unstructured/`
- Structured vs unstructured expansion
- LLM prompt variations
- Semantic drift analysis

#### 6. Voting Ensemble
- Location: `report/performance_vote/`
- Combination of multiple methods
- Voting strategies (majority, weighted)
- Failure case analysis

### Extracting Results

```python
# Load performance metrics
import json

with open('report/performance_prf/performances_extracted.json') as f:
    prf_results = json.load(f)

# Analyze specific method
method = 'gemma:27b_k5'
print(f"R@10: {prf_results[method]['R@10']}")
print(f"MRR: {prf_results[method]['MRR']}")
```

## Dependencies

### Core Libraries

- **CLIP** (`openai/clip`): Contrastive Language-Image Pre-training for shared embedding space
- **BLIP** (`Salesforce/blip-image-captioning-base`): Vision-language model for image caption generation
- **FAISS** (`faiss-cpu` or `faiss-gpu`): Facebook AI Similarity Search for efficient vector indexing and retrieval
- **Transformers** (`transformers`): Hugging Face library for loading BLIP and other vision models
- **Ollama**: Local LLM inference server for Gemma, Mistral, Llama models (must be installed separately)
- **PyTorch** (`torch`): Deep learning framework for model inference

### Python Packages

The complete dependency list is in `requirements.txt`:

```txt
# Core ML frameworks
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0

# CLIP and vision models
git+https://github.com/openai/CLIP.git
pillow>=9.0.0

# Vector search
faiss-cpu>=1.7.4  # Use faiss-gpu for GPU acceleration

# Data processing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
tqdm>=4.62.0
requests>=2.26.0
pathlib>=1.0.1

# LLM integration
ollama  # Python client for Ollama server
```

### External Requirements

**Ollama Installation** (for LLM-based methods):

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download

# Pull required models
ollama pull gemma3:4b
ollama pull gemma3:27b
ollama pull mistral:7b
ollama pull llama3.1:8b

# Verify installation
ollama list
```

**GPU Support** (recommended):
- CUDA 11.8+ for PyTorch GPU acceleration
- CUDA-compatible GPU with 8GB+ VRAM for BLIP inference
- For FAISS GPU: Install `faiss-gpu` instead of `faiss-cpu`

```bash
# Install GPU-accelerated FAISS
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Reproducibility

All experiments are reproducible with provided code and data:

1. **Fixed random seeds**: Consistent sampling across runs
2. **Versioned dependencies**: Exact package versions in `requirements.txt`
3. **Documented hyperparameters**: All settings recorded in experiment configs
4. **Raw results**: Complete experimental outputs in `report/` folders
5. **Evaluation scripts**: Automated metric computation for consistency

### Running Experiments

The project includes standalone scripts for each method in their respective report folders:

```bash
# Baseline evaluation
cd report/performance_raw/COCO
python evaluate_baseline.py

# PRF experiments with different LLMs and k values
cd report/performance_prf/COCO
python run_prf_experiments.py --llm gemma3:27b --k 5
python run_prf_experiments.py --llm mistral:7b --k 10

# Double embeddings with various alpha weights
cd report/performance_double_embeddings/Flickr
python run_hybrid_experiments.py --alpha 0.3
python run_hybrid_experiments.py --alpha 0.5
python run_hybrid_experiments.py --alpha 0.7

# Query reformulation (structured vs unstructured)
cd report/performance_structured_unstructured
python run_reformulation.py --mode structured --llm gemma3:4b
python run_reformulation.py --mode unstructured --llm mistral:7b

# IVF + LLM cluster routing
cd report/performance_ivf_llm_desc/COCO
python run_ivf_llm.py --clusters 512 --probe 10

# Voting ensemble
cd report/performance_vote
python run_voting.py --methods baseline,prf,two_embeddings

# Generate visualizations
cd visualizations/prf
python create_plots.py  # Creates recall curves, MRR comparisons, etc.
```

## Future Work

Potential extensions of this research:

1. **Cross-modal attention mechanisms**: Learn adaptive weighting between modalities
2. **Fine-tuned VLMs**: Specialize BLIP for retrieval-specific captions
3. **Efficient IVF routing**: Improve cluster-based scaling methods
4. **Multi-stage pipelines**: Combine early and late integration strategically
5. **Interactive retrieval**: User feedback loops with LLM refinement
6. **Zero-shot generalization**: Test on unseen datasets and domains

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{datalab2_2025,
  title={Evaluating LLM and VLM Integration Strategies for Text-to-Image Retrieval},
  author={[Your Name]},
  year={2025},
  journal={[Conference/Journal]},
  note={Research project on multimodal retrieval with large language models}
}
```


## Acknowledgments

- OpenAI for CLIP
- Salesforce for BLIP
- Facebook AI for FAISS
- Hugging Face for Transformers
- Ollama for local LLM inference
- MS COCO, Flickr30K, and VizWiz dataset creators

## Contact

For questions, issues, or collaboration:
- GitHub: [jeremyjrnt/Data-Laboratory-2](https://github.com/jeremyjrnt/Data-Laboratory-2)
- Issues: [GitHub Issues](https://github.com/jeremyjrnt/Data-Laboratory-2/issues)

---

**Last Updated**: November 20, 2025
