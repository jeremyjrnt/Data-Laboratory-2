"""
Tests de validation pour le module de configuration.
Vérifie que tous les paramètres sont correctement chargés et accessibles.
"""

from config.config import Config
from pathlib import Path


def test_ollama_settings():
    """Test des paramètres Ollama."""
    print("Testing Ollama Settings...")
    
    assert Config.OLLAMA_LOCAL is not None
    assert Config.OLLAMA_REMOTE_A5000 is not None
    assert Config.OLLAMA_REMOTE_A6000 is not None
    assert Config.OLLAMA_MODEL_DEFAULT is not None
    assert Config.OLLAMA_MODEL_LARGE is not None
    assert Config.OLLAMA_MODEL_MISTRAL is not None
    
    # Test méthodes helper
    assert Config.get_ollama_url('local') == Config.OLLAMA_LOCAL
    assert Config.get_ollama_url('a5000') == Config.OLLAMA_REMOTE_A5000
    assert Config.get_ollama_url('a6000') == Config.OLLAMA_REMOTE_A6000
    
    assert Config.get_ollama_model('default') == Config.OLLAMA_MODEL_DEFAULT
    assert Config.get_ollama_model('large') == Config.OLLAMA_MODEL_LARGE
    assert Config.get_ollama_model('mistral') == Config.OLLAMA_MODEL_MISTRAL
    
    print("✓ Ollama settings OK")


def test_huggingface_settings():
    """Test des paramètres HuggingFace."""
    print("Testing HuggingFace Settings...")
    
    assert Config.HF_MODEL_BLIP is not None
    assert Config.HF_MODEL_CLIP is not None
    assert Config.HF_MODEL_SENTENCE_TRANSFORMER is not None
    assert 'blip' in Config.HF_MODEL_BLIP.lower()
    assert 'clip' in Config.HF_MODEL_CLIP.lower()
    
    print("✓ HuggingFace settings OK")


def test_paths():
    """Test des chemins."""
    print("Testing Paths...")
    
    # Base directories
    assert isinstance(Config.PROJECT_ROOT, Path)
    assert isinstance(Config.DATA_DIR, Path)
    assert isinstance(Config.VECTORDB_DIR, Path)
    
    # Dataset directories
    assert isinstance(Config.COCO_DATA_DIR, Path)
    assert isinstance(Config.FLICKR_DATA_DIR, Path)
    assert isinstance(Config.VIZWIZ_DATA_DIR, Path)
    
    # Helper methods
    assert isinstance(Config.get_dataset_dir('COCO'), Path)
    assert isinstance(Config.get_images_dir('Flickr'), Path)
    assert isinstance(Config.get_metadata_path('VizWiz'), Path)
    
    print("✓ Paths OK")


def test_blip_parameters():
    """Test des paramètres BLIP."""
    print("Testing BLIP Parameters...")
    
    assert isinstance(Config.BLIP_NUM_BEAMS, int)
    assert isinstance(Config.BLIP_TOP_K, int)
    assert isinstance(Config.BLIP_TOP_P, float)
    assert isinstance(Config.BLIP_TEMPERATURE, float)
    assert isinstance(Config.BLIP_REPETITION_PENALTY, float)
    assert isinstance(Config.BLIP_MIN_LENGTH, int)
    assert isinstance(Config.BLIP_MAX_LENGTH, int)
    
    # Test helper method
    params = Config.get_blip_generation_params()
    assert isinstance(params, dict)
    assert 'num_beams' in params
    assert 'temperature' in params
    assert 'max_length' in params
    
    print("✓ BLIP parameters OK")


def test_retrieval_parameters():
    """Test des paramètres de retrieval."""
    print("Testing Retrieval Parameters...")
    
    assert isinstance(Config.DEFAULT_K, int)
    assert isinstance(Config.DEFAULT_TOP_K, int)
    assert isinstance(Config.DEFAULT_THRESHOLD, float)
    assert isinstance(Config.MAX_RETRY_ATTEMPTS, int)
    
    assert Config.DEFAULT_K > 0
    assert Config.DEFAULT_TOP_K > 0
    assert 0.0 <= Config.DEFAULT_THRESHOLD <= 1.0
    assert Config.MAX_RETRY_ATTEMPTS > 0
    
    print("✓ Retrieval parameters OK")


def test_ivf_parameters():
    """Test des paramètres IVF."""
    print("Testing IVF Parameters...")
    
    assert isinstance(Config.FAISS_NPROBE, int)
    assert isinstance(Config.IVF_CLUSTER_CALCULATION, str)
    assert isinstance(Config.IVF_ALPHA, float)
    
    assert Config.FAISS_NPROBE > 0
    assert Config.IVF_CLUSTER_CALCULATION in ['sqrt', 'fixed']
    assert 0.0 < Config.IVF_ALPHA < 1.0
    
    print("✓ IVF parameters OK")


def test_rocchio_parameters():
    """Test des paramètres Rocchio."""
    print("Testing Rocchio Parameters...")
    
    assert isinstance(Config.ROCCHIO_ALPHA, float)
    assert isinstance(Config.ROCCHIO_BETA, float)
    assert isinstance(Config.ROCCHIO_GAMMA, float)
    
    assert Config.ROCCHIO_ALPHA >= 0.0
    assert Config.ROCCHIO_BETA >= 0.0
    assert Config.ROCCHIO_GAMMA >= 0.0
    
    # Test helper method
    params = Config.get_rocchio_params()
    assert isinstance(params, dict)
    assert 'alpha' in params
    assert 'beta' in params
    assert 'gamma' in params
    
    print("✓ Rocchio parameters OK")


def test_fusion_weights():
    """Test des poids de fusion."""
    print("Testing Fusion Weights...")
    
    assert isinstance(Config.WEIGHT_CLASSIC, float)
    assert isinstance(Config.WEIGHT_BLIP, float)
    assert isinstance(Config.WEIGHT_AVERAGE, float)
    assert isinstance(Config.WEIGHT_BM25, float)
    
    # Test helper method
    weights = Config.get_fusion_weights()
    assert isinstance(weights, dict)
    assert 'classic' in weights
    assert 'blip' in weights
    assert 'average' in weights
    assert 'bm25' in weights
    
    print("✓ Fusion weights OK")


def test_standard_filenames():
    """Test des noms de fichiers standard."""
    print("Testing Standard Filenames...")
    
    assert Config.METADATA_FILENAME.endswith('.json')
    assert Config.SELECTED_IMAGES_FILENAME.endswith('.json')
    assert Config.PERFORMANCE_FILENAME.endswith('.json')
    assert Config.CLUSTER_POSITIONS_FILENAME.endswith('.json')
    
    print("✓ Standard filenames OK")


def run_all_tests():
    """Exécute tous les tests."""
    print("=" * 80)
    print("RUNNING CONFIGURATION VALIDATION TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_ollama_settings,
        test_huggingface_settings,
        test_paths,
        test_blip_parameters,
        test_retrieval_parameters,
        test_ivf_parameters,
        test_rocchio_parameters,
        test_fusion_weights,
        test_standard_filenames,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("\n✅ All tests passed! Configuration is valid.")
    else:
        print(f"\n❌ {failed} test(s) failed. Please check configuration.")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
