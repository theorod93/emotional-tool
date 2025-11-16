"""
EMOTION ANALYSIS - LOCAL INFERENCE
===================================
Purpose: Load trained models and perform inference
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from utils_local import setup_logger, load_config, preprocess_text

logger = setup_logger()

class EmotionInference:
    """
    Load trained models and perform inference on new texts

    Usage:
        inference = EmotionInference()
        results = inference.predict(["I am happy!", "This is sad"])
    """

    def __init__(self, model_dir='outputs/trained_models',
                 data_dir='outputs/preprocessed_batches'):
        """
        Initialize inference engine

        Loads:
        - TF-IDF vectorizer
        - Lexicon model
        - Naive Bayes model
        - SVM model
        - Emotion labels
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

        logger.info("Loading models for inference...")

        # Load TF-IDF vectorizer
        with open(self.data_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf = pickle.load(f)
        logger.info("  ✓ Loaded TF-IDF vectorizer")

        # Load emotion labels
        with open(self.data_dir / 'emotions.pkl', 'rb') as f:
            self.emotions = pickle.load(f)
        logger.info(f"  ✓ Loaded {len(self.emotions)} emotions")

        # Load models
        try:
            with open(self.model_dir / 'lexicon_model.pkl', 'rb') as f:
                self.lex_model = pickle.load(f)
            logger.info("  ✓ Loaded Lexicon model")
        except:
            logger.warning("  ⚠ Lexicon model not found")
            self.lex_model = None

        try:
            with open(self.model_dir / 'nb_model.pkl', 'rb') as f:
                self.nb_model = pickle.load(f)
            logger.info("  ✓ Loaded Naive Bayes model")
        except:
            logger.warning("  ⚠ Naive Bayes model not found")
            self.nb_model = None

        try:
            with open(self.model_dir / 'svm_model.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
            logger.info("  ✓ Loaded SVM model")
        except:
            logger.warning("  ⚠ SVM model not found")
            self.svm_model = None

        logger.info("✓ Inference engine ready\n")

    def preprocess(self, texts):
        """Preprocess texts"""
        if isinstance(texts, str):
            texts = [texts]

        return [preprocess_text(t) for t in texts]

    def predict(self, texts, return_proba=False, return_all_models=False):
        """
        Predict emotions for texts

        Args:
            texts: Single text or list of texts
            return_proba: Return probability distributions
            return_all_models: Return predictions from all models separately

        Returns:
            dict with predictions and optionally probabilities
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Preprocess
        texts_clean = self.preprocess(texts)

        # Vectorize
        X = self.tfidf.transform(texts_clean)

        # Get predictions from each model
        predictions = {}

        if self.lex_model:
            lex_preds = self.lex_model.predict(texts_clean)
            predictions['lexicon'] = lex_preds
            if return_proba:
                predictions['lexicon_proba'] = self.lex_model.predict_proba(texts_clean)

        if self.nb_model:
            nb_preds = self.nb_model.predict(X)
            predictions['naive_bayes'] = nb_preds
            if return_proba:
                predictions['naive_bayes_proba'] = self.nb_model.predict_proba(X)

        if self.svm_model:
            svm_preds = self.svm_model.predict(X)
            predictions['svm'] = svm_preds
            if return_proba:
                predictions['svm_proba'] = self.svm_model.predict_proba(X)

        # Ensemble (majority voting)
        ensemble_preds = []
        for i in range(len(texts)):
            votes = []
            if self.lex_model:
                votes.append(lex_preds[i])
            if self.nb_model:
                votes.append(nb_preds[i])
            if self.svm_model:
                votes.append(svm_preds[i])

            # Majority vote
            from collections import Counter
            ensemble_preds.append(Counter(votes).most_common(1)[0][0])

        predictions['ensemble'] = np.array(ensemble_preds)

        # Create results dictionary
        results = {
            'texts': texts,
            'texts_clean': texts_clean,
            'predictions': predictions['ensemble']
        }

        if return_all_models:
            results['all_models'] = predictions

        if return_proba:
            # Aggregate probabilities
            if self.nb_model and self.svm_model:
                # Average probabilities
                avg_proba = (predictions['naive_bayes_proba'] + 
                           predictions['svm_proba']) / 2
                results['probabilities'] = avg_proba
                results['confidence'] = np.max(avg_proba, axis=1)

        # If single input, return single result
        if single_input:
            results['predictions'] = results['predictions'][0]
            if 'confidence' in results:
                results['confidence'] = results['confidence'][0]

        return results

    def predict_with_explanation(self, text):
        """
        Predict with detailed explanation

        Returns predictions from all models plus ensemble
        """
        result = self.predict([text], return_all_models=True, return_proba=True)

        print(f"\n{'='*80}")
        print(f"TEXT: {text}")
        print(f"{'='*80}")
        print(f"\nCleaned: {result['texts_clean'][0]}")
        print(f"\nPREDICTIONS:")
        print(f"  Lexicon:     {result['all_models']['lexicon'][0]}")
        print(f"  Naive Bayes: {result['all_models']['naive_bayes'][0]}")
        print(f"  SVM:         {result['all_models']['svm'][0]}")
        print(f"  ─────────────")
        print(f"  Ensemble:    {result['predictions']}")

        if 'confidence' in result:
            print(f"\nConfidence: {result['confidence']:.3f}")

        if 'probabilities' in result:
            print(f"\nProbability Distribution:")
            probs = result['probabilities'][0]
            for i, emotion in enumerate(self.emotions):
                print(f"  {emotion:15s}: {probs[i]:.3f}")

        print(f"{'='*80}\n")

        return result

def demo():
    """Demo inference on example texts"""
    print("\n" + "="*80)
    print("EMOTION ANALYSIS - INFERENCE DEMO")
    print("="*80 + "\n")

    # Initialize inference
    inference = EmotionInference()

    # Example texts
    test_texts = [
        "I am so happy and excited about this wonderful news!",
        "This makes me very angry and frustrated.",
        "I feel sad and disappointed about the situation.",
        "I'm afraid something bad might happen.",
        "What a beautiful surprise this is!",
        "I trust that everything will work out fine."
    ]

    print("BATCH PREDICTION:")
    print("-" * 80)
    results = inference.predict(test_texts)

    for i, text in enumerate(test_texts):
        print(f"\n{i+1}. {text}")
        print(f"   → Emotion: {results['predictions'][i]}")

    print("\n\n" + "="*80)
    print("DETAILED PREDICTION WITH EXPLANATION:")
    print("="*80)

    # Detailed prediction for one text
    inference.predict_with_explanation(test_texts[0])

if __name__ == '__main__':
    demo()
