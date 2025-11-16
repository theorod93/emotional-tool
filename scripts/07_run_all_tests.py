"""
RUN ALL TESTS - COMPLETE PIPELINE
==================================
Purpose: Execute entire pipeline from data creation to inference
Date: November 2025
"""

import sys
import time
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def run_pipeline():
    """Execute complete testing pipeline"""

    print_header("EMOTION ANALYSIS - LOCAL TESTING PIPELINE")

    start_time = time.time()

    # =========================================================================
    # STEP 1: Check dependencies
    # =========================================================================
    print_header("STEP 1/5: CHECKING DEPENDENCIES")

    try:
        import pandas
        import numpy
        import sklearn
        import nltk
        import yaml
        print("✓ All core dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements_local.txt")
        return False

    # Check NLTK data
    try:
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        stopwords.words('english')
        print("✓ NLTK data available")
    except LookupError:
        print("⚠ Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✓ NLTK data downloaded")

    # =========================================================================
    # STEP 2: Create test data
    # =========================================================================
    print_header("STEP 2/5: CREATING TEST DATA")

    # Check if data already exists
    if Path('data/test_sample.csv').exists():
        print("✓ Test data already exists (data/test_sample.csv)")
        print("  Using existing file")
    else:
        print("Creating synthetic test data...")
        from create_test_data import create_test_dataset
        create_test_dataset(n_samples=1000)

    # Check lexicon
    if not Path('data/Emotion_Lexicon.csv').exists():
        print("\n✗ ERROR: Emotion_Lexicon.csv not found!")
        print("  Please place Emotion_Lexicon.csv in data/ folder")
        return False
    else:
        print("✓ Emotion lexicon found")

    # =========================================================================
    # STEP 3: Preprocessing
    # =========================================================================
    print_header("STEP 3/5: PREPROCESSING DATA")

    try:
        from preprocess_local import LocalPreprocessor
        preprocessor = LocalPreprocessor()
        preprocess_results = preprocessor.run()

        print("\n✓ Preprocessing complete!")
        print(f"  Training samples: {preprocess_results['n_train']}")
        print(f"  Validation samples: {preprocess_results['n_val']}")
        print(f"  Test samples: {preprocess_results['n_test']}")
        print(f"  Features: {preprocess_results['n_features']}")
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # STEP 4: Training with Nested CV
    # =========================================================================
    print_header("STEP 4/5: TRAINING MODELS (NESTED CV)")

    print("⏱ This may take 5-10 minutes on CPU...\n")

    try:
        from train_local import LocalTrainer
        trainer = LocalTrainer()
        cv_summary = trainer.run()

        print("\n✓ Training complete!")
        print("  Models saved to outputs/trained_models/")
        print("  CV results saved to outputs/cv_results/")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # STEP 5: Inference Demo
    # =========================================================================
    print_header("STEP 5/5: INFERENCE DEMO")

    try:
        from inference_local import EmotionInference
        inference = EmotionInference()

        # Test texts
        test_texts = [
            "I am so happy and excited!",
            "This makes me very angry.",
            "I feel sad and disappointed."
        ]

        print("Testing inference on sample texts:\n")
        results = inference.predict(test_texts)

        for i, text in enumerate(test_texts):
            print(f"{i+1}. {text}")
            print(f"   → Predicted emotion: {results['predictions'][i]}\n")

        print("✓ Inference working correctly!")
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print_header("PIPELINE COMPLETE!")

    print(f"✓ All steps completed successfully!")
    print(f"⏱ Total time: {minutes}m {seconds}s\n")

    print("Generated files:")
    print("  • data/test_sample.csv - Test dataset")
    print("  • outputs/preprocessed_batches/ - Processed data")
    print("  • outputs/trained_models/ - Trained models")
    print("  • outputs/cv_results/ - Cross-validation results")
    print("  • logs/training_log.txt - Detailed logs\n")

    print("Next steps:")
    print("  1. Review CV results: cat outputs/cv_results/cv_results.json")
    print("  2. Test inference: python 05_inference_local.py")
    print("  3. Check logs: cat logs/training_log.txt")
    print("  4. Deploy to AWS: Follow AWS_SETUP_GUIDE.md\n")

    print("="*80)

    return True

if __name__ == '__main__':
    success = run_pipeline()
    sys.exit(0 if success else 1)
