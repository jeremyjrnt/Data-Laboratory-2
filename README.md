# RAG Project with FAISS, Streamlit, Ollama & HuggingFace

This project implements a Retrieval-Augmented Generation (RAG) system using:
- **FAISS** for vector similarity search
- **Streamlit** for the web interface
- **Ollama** for local LLM inference
- **HuggingFace** for embeddings and transformers

## Setup

### 1. Clone and navigate to the project
```bash
git clone <your-repo>
cd DataLab2Project
```

### 2. Create and activate virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
cd .\.venv\Scripts\
.\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your specific settings
```

### 5. Install and run Ollama
```bash
# Install Ollama from https://ollama.ai
ollama pull llama2
ollama serve
```

## Project Structure

```
DataLab2Project/
├── src/
│   ├── embeddings/        # Embedding generation modules
│   ├── retrieval/         # FAISS indexing and retrieval
│   ├── llm/               # Ollama integration
│   └── streamlit_app/     # Streamlit interface
├── data/
│   ├── raw/               # Original documents
│   └── processed/         # Processed and chunked data
├── models/                # Saved models and indices
├── config/                # Configuration files
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Usage

1. **Prepare your data**: Place documents in `data/raw/`
2. **Build index**: Run the indexing script to create FAISS index
3. **Launch app**: Start the Streamlit application
4. **Query**: Ask questions about your documents

## Technologies

- **FAISS**: Fast similarity search and clustering
- **Streamlit**: Interactive web applications
- **Ollama**: Local LLM inference
- **HuggingFace Transformers**: Pre-trained models and tokenizers
- **Sentence Transformers**: Semantic sentence embeddings
- **LangChain**: LLM application framework
