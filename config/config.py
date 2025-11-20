"""
Configuration module for the RAG project.
Loads all settings from .env file and provides easy access throughout the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Central configuration class for all project settings."""
    
    # === Ollama Settings ===
    OLLAMA_REMOTE_A5000 = os.getenv('OLLAMA_REMOTE_A5000', 'http://100.64.0.7:11434/api/generate')
    OLLAMA_REMOTE_A6000 = os.getenv('OLLAMA_REMOTE_A6000', 'http://100.64.0.9:11434/api/generate')
    OLLAMA_LOCAL = os.getenv('OLLAMA_LOCAL', 'http://localhost:11434/api/generate')
    OLLAMA_MODEL_DEFAULT = os.getenv('OLLAMA_MODEL_DEFAULT', 'gemma3:4b')
    OLLAMA_MODEL_LARGE = os.getenv('OLLAMA_MODEL_LARGE', 'gemma3:27b')
    OLLAMA_MODEL_MISTRAL = os.getenv('OLLAMA_MODEL_MISTRAL', 'mistral:7b')
    
    # === HuggingFace Settings ===
    HF_TOKEN = os.getenv('HF_TOKEN', 'your_huggingface_token_here')
    HF_MODEL_SENTENCE_TRANSFORMER = os.getenv('HF_MODEL_SENTENCE_TRANSFORMER', 'sentence-transformers/all-MiniLM-L6-v2')
    HF_MODEL_BLIP = os.getenv('HF_MODEL_BLIP', 'Salesforce/blip-image-captioning-large')
    HF_MODEL_CLIP = os.getenv('HF_MODEL_CLIP', 'openai/clip-vit-base-patch32')
    HF_MODEL_CLIP_LARGE = os.getenv('HF_MODEL_CLIP_LARGE', 'openai/clip-vit-large-patch14')
    
    # === FAISS Settings ===
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', './models/faiss_index.index')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '384'))
    
    # === Streamlit Settings ===
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', '8501'))
    STREAMLIT_HOST = os.getenv('STREAMLIT_HOST', 'localhost')
    
    # === Base Directories ===
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / os.getenv('DATA_DIR', 'data')
    VECTORDB_DIR = PROJECT_ROOT / os.getenv('VECTORDB_DIR', 'VectorDBs')
    REPORT_DIR = PROJECT_ROOT / os.getenv('REPORT_DIR', 'report')
    CONFIG_DIR = PROJECT_ROOT / os.getenv('CONFIG_DIR', 'config')
    
    # === Dataset Directories ===
    COCO_DATA_DIR = PROJECT_ROOT / os.getenv('COCO_DATA_DIR', 'data/COCO')
    FLICKR_DATA_DIR = PROJECT_ROOT / os.getenv('FLICKR_DATA_DIR', 'data/Flickr')
    VIZWIZ_DATA_DIR = PROJECT_ROOT / os.getenv('VIZWIZ_DATA_DIR', 'data/VizWiz')
    
    # === Image Directories ===
    COCO_IMAGES_DIR = PROJECT_ROOT / os.getenv('COCO_IMAGES_DIR', 'data/COCO/images')
    FLICKR_IMAGES_DIR = PROJECT_ROOT / os.getenv('FLICKR_IMAGES_DIR', 'data/Flickr/images')
    VIZWIZ_IMAGES_DIR = PROJECT_ROOT / os.getenv('VIZWIZ_IMAGES_DIR', 'data/VizWiz/images')
    
    # === Standard Filenames ===
    METADATA_FILENAME = os.getenv('METADATA_FILENAME', 'metadata.json')
    SELECTED_IMAGES_FILENAME = os.getenv('SELECTED_IMAGES_FILENAME', 'selected_1000.json')
    PERFORMANCE_FILENAME = os.getenv('PERFORMANCE_FILENAME', 'performance.json')
    CLUSTER_POSITIONS_FILENAME = os.getenv('CLUSTER_POSITIONS_FILENAME', 'cluster_positions.json')
    CORPUS_FILENAME = os.getenv('CORPUS_FILENAME', 'corpus.json')
    
    # === Processing Settings ===
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2048'))
    MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', '40'))
    
    # === Model Generation Parameters ===
    BLIP_NUM_BEAMS = int(os.getenv('BLIP_NUM_BEAMS', '5'))
    BLIP_TOP_K = int(os.getenv('BLIP_TOP_K', '50'))
    BLIP_TOP_P = float(os.getenv('BLIP_TOP_P', '0.95'))
    BLIP_TEMPERATURE = float(os.getenv('BLIP_TEMPERATURE', '0.8'))
    BLIP_REPETITION_PENALTY = float(os.getenv('BLIP_REPETITION_PENALTY', '1.5'))
    BLIP_MIN_LENGTH = int(os.getenv('BLIP_MIN_LENGTH', '30'))
    BLIP_MAX_LENGTH = int(os.getenv('BLIP_MAX_LENGTH', '150'))
    
    # === FAISS IVF Settings ===
    FAISS_NPROBE = int(os.getenv('FAISS_NPROBE', '10'))
    IVF_CLUSTER_CALCULATION = os.getenv('IVF_CLUSTER_CALCULATION', 'sqrt')
    IVF_ALPHA = float(os.getenv('IVF_ALPHA', '0.5'))
    
    # === LLM Generation Parameters ===
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.1'))
    LLM_TOP_P = float(os.getenv('LLM_TOP_P', '0.9'))
    LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '200'))
    
    # === Retrieval Settings ===
    DEFAULT_K = int(os.getenv('DEFAULT_K', '100'))
    DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', '10'))
    DEFAULT_THRESHOLD = float(os.getenv('DEFAULT_THRESHOLD', '0.0'))
    MAX_RETRY_ATTEMPTS = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
    
    # === Rocchio PRF Parameters ===
    ROCCHIO_ALPHA = float(os.getenv('ROCCHIO_ALPHA', '1.0'))
    ROCCHIO_BETA = float(os.getenv('ROCCHIO_BETA', '0.75'))
    ROCCHIO_GAMMA = float(os.getenv('ROCCHIO_GAMMA', '0.0'))
    
    # === BM25 Parameters ===
    BM25_K1 = float(os.getenv('BM25_K1', '1.2'))
    BM25_B = float(os.getenv('BM25_B', '0.75'))
    
    # === Two Embeddings Fusion Weights ===
    WEIGHT_CLASSIC = float(os.getenv('WEIGHT_CLASSIC', '0.5'))
    WEIGHT_BLIP = float(os.getenv('WEIGHT_BLIP', '0.5'))
    WEIGHT_AVERAGE = float(os.getenv('WEIGHT_AVERAGE', '0.5'))
    WEIGHT_BM25 = float(os.getenv('WEIGHT_BM25', '0.5'))
    
    # === Evaluation Parameters ===
    EVALUATION_CASES_PER_RANK = int(os.getenv('EVALUATION_CASES_PER_RANK', '1000'))
    
    @classmethod
    def get_dataset_dir(cls, dataset_name: str) -> Path:
        """Get the data directory for a specific dataset."""
        dataset_map = {
            'COCO': cls.COCO_DATA_DIR,
            'Flickr': cls.FLICKR_DATA_DIR,
            'VizWiz': cls.VIZWIZ_DATA_DIR
        }
        return dataset_map.get(dataset_name, cls.DATA_DIR / dataset_name)
    
    @classmethod
    def get_images_dir(cls, dataset_name: str) -> Path:
        """Get the images directory for a specific dataset."""
        images_map = {
            'COCO': cls.COCO_IMAGES_DIR,
            'Flickr': cls.FLICKR_IMAGES_DIR,
            'VizWiz': cls.VIZWIZ_IMAGES_DIR
        }
        return images_map.get(dataset_name, cls.DATA_DIR / dataset_name / 'images')
    
    @classmethod
    def get_metadata_path(cls, dataset_name: str) -> Path:
        """Get the metadata file path for a specific dataset."""
        return cls.get_dataset_dir(dataset_name) / f"{dataset_name.lower()}_{cls.METADATA_FILENAME}"
    
    @classmethod
    def get_selected_images_path(cls, dataset_name: str) -> Path:
        """Get the selected images file path for a specific dataset."""
        return cls.get_dataset_dir(dataset_name) / cls.SELECTED_IMAGES_FILENAME
    
    @classmethod
    def get_vectordb_metadata_path(cls, db_name: str) -> Path:
        """Get the VectorDB metadata file path."""
        return cls.VECTORDB_DIR / f"{db_name}_metadata.json"
    
    @classmethod
    def get_vectordb_index_path(cls, db_name: str) -> Path:
        """Get the VectorDB index file path."""
        return cls.VECTORDB_DIR / f"{db_name}.index"
    
    @classmethod
    def get_ollama_url(cls, server: str = 'local') -> str:
        """
        Get Ollama API URL for specific server.
        
        Args:
            server: 'local', 'a5000', or 'a6000'
        """
        server_map = {
            'local': cls.OLLAMA_LOCAL,
            'a5000': cls.OLLAMA_REMOTE_A5000,
            'a6000': cls.OLLAMA_REMOTE_A6000
        }
        return server_map.get(server.lower(), cls.OLLAMA_LOCAL)
    
    @classmethod
    def get_ollama_model(cls, model_type: str = 'default') -> str:
        """
        Get Ollama model name.
        
        Args:
            model_type: 'default', 'large', or 'mistral'
        """
        model_map = {
            'default': cls.OLLAMA_MODEL_DEFAULT,
            'large': cls.OLLAMA_MODEL_LARGE,
            'mistral': cls.OLLAMA_MODEL_MISTRAL
        }
        return model_map.get(model_type.lower(), cls.OLLAMA_MODEL_DEFAULT)
    
    @classmethod
    def get_blip_generation_params(cls) -> dict:
        """Get BLIP generation parameters as a dictionary."""
        return {
            'max_new_tokens': cls.MAX_NEW_TOKENS,
            'num_beams': cls.BLIP_NUM_BEAMS,
            'top_k': cls.BLIP_TOP_K,
            'top_p': cls.BLIP_TOP_P,
            'temperature': cls.BLIP_TEMPERATURE,
            'repetition_penalty': cls.BLIP_REPETITION_PENALTY,
            'min_length': cls.BLIP_MIN_LENGTH,
            'max_length': cls.BLIP_MAX_LENGTH
        }
    
    @classmethod
    def get_rocchio_params(cls) -> dict:
        """Get Rocchio PRF parameters as a dictionary."""
        return {
            'alpha': cls.ROCCHIO_ALPHA,
            'beta': cls.ROCCHIO_BETA,
            'gamma': cls.ROCCHIO_GAMMA
        }
    
    @classmethod
    def get_fusion_weights(cls) -> dict:
        """Get embedding fusion weights as a dictionary."""
        return {
            'classic': cls.WEIGHT_CLASSIC,
            'blip': cls.WEIGHT_BLIP,
            'average': cls.WEIGHT_AVERAGE,
            'bm25': cls.WEIGHT_BM25
        }


# Create a singleton instance for easy import
config = Config()


if __name__ == "__main__":
    # Display configuration for debugging
    print("=== RAG Project Configuration ===\n")
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"\n--- Ollama Settings ---")
    print(f"Local URL: {Config.OLLAMA_LOCAL}")
    print(f"Remote A5000: {Config.OLLAMA_REMOTE_A5000}")
    print(f"Remote A6000: {Config.OLLAMA_REMOTE_A6000}")
    print(f"Default Model: {Config.OLLAMA_MODEL_DEFAULT}")
    print(f"\n--- HuggingFace Models ---")
    print(f"BLIP: {Config.HF_MODEL_BLIP}")
    print(f"CLIP: {Config.HF_MODEL_CLIP}")
    print(f"\n--- Data Directories ---")
    print(f"Data: {Config.DATA_DIR}")
    print(f"VectorDB: {Config.VECTORDB_DIR}")
    print(f"COCO: {Config.COCO_DATA_DIR}")
    print(f"Flickr: {Config.FLICKR_DATA_DIR}")
    print(f"VizWiz: {Config.VIZWIZ_DATA_DIR}")
    print(f"\n--- Retrieval Settings ---")
    print(f"Default K: {Config.DEFAULT_K}")
    print(f"Default Top-K: {Config.DEFAULT_TOP_K}")
    print(f"Default Threshold: {Config.DEFAULT_THRESHOLD}")
    print(f"Max Retry Attempts: {Config.MAX_RETRY_ATTEMPTS}")
    print(f"\n--- BLIP Generation Parameters ---")
    blip_params = Config.get_blip_generation_params()
    for key, value in blip_params.items():
        print(f"{key}: {value}")
    print(f"\n--- FAISS IVF Settings ---")
    print(f"nprobe: {Config.FAISS_NPROBE}")
    print(f"Cluster Calculation: {Config.IVF_CLUSTER_CALCULATION}")
    print(f"Alpha: {Config.IVF_ALPHA}")
    print(f"\n--- Rocchio PRF Parameters ---")
    rocchio_params = Config.get_rocchio_params()
    for key, value in rocchio_params.items():
        print(f"{key}: {value}")
    print(f"\n--- Fusion Weights ---")
    fusion_weights = Config.get_fusion_weights()
    for key, value in fusion_weights.items():
        print(f"{key}: {value}")
