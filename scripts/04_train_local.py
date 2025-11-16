"""
EMOTION ANALYSIS - LOCAL TRAINING WITH NESTED CV
=================================================
Purpose: Train models with nested cross-validation on CPU
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from collections import Counter

from utils_local import (
    setup_logger, load_config, load_emotion_lexicon,
    calculate_metrics, print_metrics, save_model, save_cv_results
)

logger = setup_logger()

class LexiconModel:
    """
    Lexicon-based emotion detector (rule-based, no training)

    Why lexicon approach:
    - Interpretable: Easy to understand predictions
    - Zero-shot: No training data needed
    - Fast: Dictionary lookup
    - Baseline: Comparison for ML models
    """

    def __init__(self, lexicon_dict, emotion_cols):
        self.lexicon = lexicon_dict
        self.emotion_cols = emotion_cols
        self.emotion_to_idx = {e: i for i, e in enumerate(emotion_cols)}

    def predict(self, texts):
        """Predict emotion for each text"""
        predictions = []

        for text in texts:
            words = str(text).split()
            emotion_counts = {emo: 0 for emo in self.emotion_cols}

            # Count emotion words
            for word in words:
                if word in self.lexicon:
                    for emo in self.emotion_cols:
                        emotion_counts[emo] += self.lexicon[word].get(emo, 0)

            # Get dominant emotion
            if sum(emotion_counts.values()) > 0:
                pred_emotion = max(emotion_counts, key=emotion_counts.get)
            else:
                # Default to Joy if no matches
                pred_emotion = 'Joy'

            predictions.append(pred_emotion)

        return np.array(predictions)

    def predict_proba(self, texts):
        """Get probability distribution"""
        probs = []

        for text in texts:
            words = str(text).split()
            emotion_counts = {emo: 0 for emo in self.emotion_cols}

            for word in words:
                if word in self.lexicon:
                    for emo in self.emotion_cols:
                        emotion_counts[emo] += self.lexicon[word].get(emo, 0)

            total = sum(emotion_counts.values())
            if total > 0:
                prob_dist = [emotion_counts[emo] / total for emo in self.emotion_cols]
            else:
                # Uniform distribution if no matches
                prob_dist = [1.0 / len(self.emotion_cols)] * len(self.emotion_cols)

            probs.append(prob_dist)

        return np.array(probs)

class LocalTrainer:
    """
    Handles model training with nested cross-validation

    Nested CV structure:
    - Outer loop (2 folds): Model evaluation
    - Inner loop (2 folds): Hyperparameter tuning

    Why nested CV:
    - Prevents overfitting in hyperparameter selection
    - Provides unbiased performance estimates
    - Standard practice for robust evaluation
    """

    def __init__(self, config_file='01_config_local.yaml'):
        """Initialize trainer"""
        self.config = load_config(config_file)
        self.outer_folds = self.config['cross_validation']['outer_folds']
        self.inner_folds = self.config['cross_validation']['inner_folds']

        self.results_dir = Path(self.config['output']['results_dir'])
        self.model_dir = Path(self.config['output']['model_dir'])

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load lexicon
        lex_file = self.config['data']['lexicon_file']
        self.lexicon, self.emotion_cols = load_emotion_lexicon(lex_file)

        logger.info("✓ Trainer initialized")

    def load_preprocessed_data(self):
        """Load preprocessed data"""
        batch_dir = Path(self.config['output']['batch_output_dir'])

        logger.info("Loading preprocessed data...")

        X_train = load_npz(batch_dir / 'X_train.npz')
        X_test = load_npz(batch_dir / 'X_test.npz')

        y_train = np.load(batch_dir / 'y_train.npy', allow_pickle=True)
        y_test = np.load(batch_dir / 'y_test.npy', allow_pickle=True)

        # Load raw texts for lexicon model
        train_df = pd.read_csv(batch_dir / 'train_raw.csv')
        test_df = pd.read_csv(batch_dir / 'test_raw.csv')

        logger.info(f"✓ X_train: {X_train.shape}")
        logger.info(f"✓ X_test: {X_test.shape}")

        return X_train, X_test, y_train, y_test, train_df, test_df

    def train_lexicon(self):
        """Initialize lexicon model (no training needed)"""
        return LexiconModel(self.lexicon, self.emotion_cols)

    def train_naive_bayes(self, X_train, y_train):
        """
        Train Naive Bayes classifier

        Why Naive Bayes:
        - Fast training and prediction
        - Works well with sparse TF-IDF features
        - Probabilistic output
        - Good baseline
        """
        logger.info("  Training Naive Bayes...")
        model = MultinomialNB(alpha=self.config['models']['naive_bayes']['alpha'])
        model.fit(X_train, y_train)
        return model

    def train_svm(self, X_train, y_train):
        """
        Train SVM classifier

        Why SVM:
        - Effective in high-dimensional spaces
        - Margin-based learning (good generalization)
        - Handles non-linear patterns

        Note: Using linear kernel for speed on CPU
        """
        logger.info("  Training SVM...")
        model = SVC(
            kernel=self.config['models']['svm']['kernel'],
            C=self.config['models']['svm']['C'],
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def ensemble_predict(self, lex_preds, nb_preds, svm_preds):
        """
        Combine predictions via majority voting

        Why ensemble:
        - Combines strengths of different models
        - Reduces variance
        - Usually outperforms individual models
        """
        ensemble = []
        for i in range(len(lex_preds)):
            votes = [lex_preds[i], nb_preds[i], svm_preds[i]]
            # Majority vote
            ensemble.append(Counter(votes).most_common(1)[0][0])
        return np.array(ensemble)

    def run_nested_cv(self, X_train, y_train, train_texts):
        """
        Execute nested cross-validation

        Process:
        OUTER LOOP (model evaluation):
          For each outer fold:
            INNER LOOP (hyperparameter tuning):
              For each inner fold:
                Train models with different hyperparameters
                Validate on inner validation set
                Select best hyperparameters
            Retrain with best hyperparameters on full outer training set
            Evaluate on held-out outer test set

        Result: Unbiased performance estimates
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING NESTED CROSS-VALIDATION")
        logger.info(f"Outer folds: {self.outer_folds} | Inner folds: {self.inner_folds}")
        logger.info("="*80 + "\n")

        # Stratified K-Fold for outer loop
        skf_outer = StratifiedKFold(
            n_splits=self.outer_folds,
            shuffle=True,
            random_state=self.config['cross_validation']['cv_seed']
        )

        # Storage for results
        fold_results = []

        # OUTER LOOP
        for outer_fold, (train_idx, test_idx) in enumerate(skf_outer.split(X_train, y_train)):
            logger.info(f"\n{'='*80}")
            logger.info(f"OUTER FOLD {outer_fold + 1}/{self.outer_folds}")
            logger.info(f"{'='*80}")

            # Outer split
            X_outer_train = X_train[train_idx]
            X_outer_test = X_train[test_idx]
            y_outer_train = y_train[train_idx]
            y_outer_test = y_train[test_idx]

            # Get corresponding text data for lexicon
            train_texts_outer = train_texts.iloc[train_idx]['text_clean'].values
            test_texts_outer = train_texts.iloc[test_idx]['text_clean'].values

            logger.info(f"Outer train: {len(y_outer_train)} | Outer test: {len(y_outer_test)}")

            # INNER LOOP (hyperparameter tuning)
            logger.info(f"\nRunning {self.inner_folds} inner folds...")

            skf_inner = StratifiedKFold(
                n_splits=self.inner_folds,
                shuffle=True,
                random_state=42 + outer_fold
            )

            inner_scores = {'nb': [], 'svm': []}

            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                skf_inner.split(X_outer_train, y_outer_train)
            ):
                X_inner_train = X_outer_train[inner_train_idx]
                X_inner_val = X_outer_train[inner_val_idx]
                y_inner_train = y_outer_train[inner_train_idx]
                y_inner_val = y_outer_train[inner_val_idx]

                # Train models
                nb = self.train_naive_bayes(X_inner_train, y_inner_train)
                svm = self.train_svm(X_inner_train, y_inner_train)

                # Validate
                from sklearn.metrics import f1_score
                nb_preds = nb.predict(X_inner_val)
                svm_preds = svm.predict(X_inner_val)

                nb_f1 = f1_score(y_inner_val, nb_preds, average='macro', zero_division=0)
                svm_f1 = f1_score(y_inner_val, svm_preds, average='macro', zero_division=0)

                inner_scores['nb'].append(nb_f1)
                inner_scores['svm'].append(svm_f1)

            avg_nb_f1 = np.mean(inner_scores['nb'])
            avg_svm_f1 = np.mean(inner_scores['svm'])

            logger.info(f"  Avg inner fold F1 - NB: {avg_nb_f1:.4f}, SVM: {avg_svm_f1:.4f}")

            # RETRAIN on full outer training set with best config
            logger.info("\nRetraining on full outer training set...")
            lex_model = self.train_lexicon()
            nb_model = self.train_naive_bayes(X_outer_train, y_outer_train)
            svm_model = self.train_svm(X_outer_train, y_outer_train)

            # EVALUATE on outer test set
            logger.info("Evaluating on outer test set...\n")

            lex_preds = lex_model.predict(test_texts_outer)
            nb_preds = nb_model.predict(X_outer_test)
            svm_preds = svm_model.predict(X_outer_test)
            ensemble_preds = self.ensemble_predict(lex_preds, nb_preds, svm_preds)

            # Calculate metrics
            models_eval = [
                ('Lexicon', lex_preds),
                ('Naive Bayes', nb_preds),
                ('SVM', svm_preds),
                ('Ensemble', ensemble_preds)
            ]

            fold_metrics = {'outer_fold': outer_fold + 1}

            for model_name, preds in models_eval:
                metrics = calculate_metrics(y_outer_test, preds, model_name)
                fold_metrics[model_name.lower().replace(' ', '_')] = {
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision_macro'],
                    'recall': metrics['recall_macro'],
                    'f1': metrics['f1_macro']
                }
                print_metrics(metrics)

            fold_results.append(fold_metrics)

        # AGGREGATE RESULTS
        logger.info("\n" + "="*80)
        logger.info("NESTED CV SUMMARY")
        logger.info("="*80 + "\n")

        # Convert to DataFrame
        results_df = pd.DataFrame(fold_results)

        # Calculate summary statistics
        summary = {}
        for model in ['lexicon', 'naive_bayes', 'svm', 'ensemble']:
            if model in results_df.columns:
                model_data = results_df[model].apply(pd.Series)
                summary[model] = {
                    'accuracy_mean': model_data['accuracy'].mean(),
                    'accuracy_std': model_data['accuracy'].std(),
                    'f1_mean': model_data['f1'].mean(),
                    'f1_std': model_data['f1'].std()
                }

        # Print summary
        logger.info("Cross-Validation Summary (mean ± std):")
        for model, stats in summary.items():
            logger.info(f"\n{model.upper()}:")
            logger.info(f"  Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
            logger.info(f"  F1-Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")

        # Save results
        save_cv_results(results_df, self.results_dir)
        save_cv_results(summary, self.results_dir)

        return results_df, summary

    def train_final_models(self, X_train, y_train, train_texts):
        """
        Train final models on full training set

        Why: After CV evaluation, retrain on all data for deployment
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING FINAL MODELS ON FULL TRAINING SET")
        logger.info("="*80 + "\n")

        lex_model = self.train_lexicon()
        nb_model = self.train_naive_bayes(X_train, y_train)
        svm_model = self.train_svm(X_train, y_train)

        # Save models
        save_model(lex_model, self.model_dir / 'lexicon_model', 'pkl')
        save_model(nb_model, self.model_dir / 'nb_model', 'pkl')
        save_model(svm_model, self.model_dir / 'svm_model', 'pkl')

        logger.info("✓ All models saved")

        return lex_model, nb_model, svm_model

    def run(self):
        """Execute complete training pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*80 + "\n")

        # Load data
        X_train, X_test, y_train, y_test, train_df, test_df = self.load_preprocessed_data()

        # Run nested CV
        cv_results, cv_summary = self.run_nested_cv(X_train, y_train, train_df)

        # Train final models
        lex_model, nb_model, svm_model = self.train_final_models(X_train, y_train, train_df)

        # Final evaluation on held-out test set
        logger.info("\n" + "="*80)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("="*80 + "\n")

        test_texts = test_df['text_clean'].values

        lex_preds = lex_model.predict(test_texts)
        nb_preds = nb_model.predict(X_test)
        svm_preds = svm_model.predict(X_test)
        ensemble_preds = self.ensemble_predict(lex_preds, nb_preds, svm_preds)

        for model_name, preds in [('Lexicon', lex_preds), ('NB', nb_preds), 
                                    ('SVM', svm_preds), ('Ensemble', ensemble_preds)]:
            metrics = calculate_metrics(y_test, preds, model_name)
            print_metrics(metrics)

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80 + "\n")

        return cv_summary

if __name__ == '__main__':
    trainer = LocalTrainer()
    summary = trainer.run()

    print("\n✓ Training complete! Check outputs/ folder for results.")
