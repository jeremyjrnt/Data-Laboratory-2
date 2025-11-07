#!/usr/bin/env python3
"""
Test script for PRF performance evaluation
Quick validation that the system is working correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.performance.performance_prf import PerformancePRFEvaluator


def test_evaluator_initialization():
    """Test that the evaluator can be initialized."""
    print("=" * 80)
    print("Testing PerformancePRFEvaluator initialization...")
    print("=" * 80)
    
    try:
        evaluator = PerformancePRFEvaluator("COCO")
        print(f"✅ Evaluator initialized successfully")
        print(f"   Dataset: {evaluator.dataset_name}")
        print(f"   VectorDB: {evaluator.vectordb_name}")
        print(f"   Test images: {len(evaluator.selected_images)}")
        print(f"   LLM models: {evaluator.llm_models}")
        print(f"   Rocchio configs: {len(evaluator.rocchio_params)}")
        print()
        
        # Display Rocchio configurations
        print("Rocchio parameter configurations:")
        for i, (alpha, beta, gamma) in enumerate(evaluator.rocchio_params, 1):
            print(f"  {i}. α={alpha}, β={beta}, γ={gamma}")
        
        print()
        print("=" * 80)
        print("✅ All checks passed!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retriever_with_rocchio_params():
    """Test that the retriever accepts Rocchio parameters."""
    print("\n" + "=" * 80)
    print("Testing ImageRetriever with custom Rocchio parameters...")
    print("=" * 80)
    
    try:
        from src.retrieval.retriever_prf import ImageRetriever
        
        # Test with default parameters
        print("\n1. Testing with default parameters...")
        retriever1 = ImageRetriever(
            vectordb_name="COCO_VectorDB",
            llm_model="mistral:7b"
        )
        print(f"   ✅ Default: α={retriever1.alpha}, β={retriever1.beta}, γ={retriever1.gamma}")
        
        # Test with custom parameters
        print("\n2. Testing with custom parameters...")
        retriever2 = ImageRetriever(
            vectordb_name="COCO_VectorDB",
            llm_model="mistral:7b",
            alpha=1.5,
            beta=0.5,
            gamma=0.1
        )
        print(f"   ✅ Custom: α={retriever2.alpha}, β={retriever2.beta}, γ={retriever2.gamma}")
        
        print()
        print("=" * 80)
        print("✅ Retriever parameter test passed!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪" * 40)
    print("PRF Performance Evaluation - System Test")
    print("🧪" * 40 + "\n")
    
    tests = [
        ("Evaluator Initialization", test_evaluator_initialization),
        ("Retriever Rocchio Parameters", test_retriever_with_rocchio_params),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - System is ready for evaluation!")
    else:
        print("⚠️  SOME TESTS FAILED - Please check the errors above")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
