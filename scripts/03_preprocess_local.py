"""
EMOTION ANALYSIS - LOCAL PREPROCESSING
=======================================
Purpose: Preprocess data for local CPU training
Date: November 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils_local import (
    setup_logger, load_config, preprocess_text,
    load_emotion_lexicon, extract_emotion_label
)

logger = setup_logger()

class LocalPreprocessor:
    """
    Handles data preprocessing for local testing

    Steps:
    1. Load data
    2. Clean text
    3. Extract emotion labels
    4. Train/val/test split (stratified)
    5. TF-IDF vectorization
    6. Save processed data
    """

    def __init__(self, config_file='01_config_local.yaml'):
        """Initialize preprocessor"""
        self.config = load_config(config_file)
        self.output_dir = Path(self.config['output']['batch_output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load lexicon
        lex_file = self.config['data']['lexicon_file']
        self.lexicon, self.emotion_cols = load_emotion_lexicon(lex_file)

        logger.info("✓ Preprocessor initialized")

    def load_data(self):
        """Load dataset from CSV"""
        input_file = self.config['data']['input_file']
        logger.info(f"Loading data from {input_file}")

        try:
            df = pd.read_csv(input_file)
            logger.info(f"✓ Loaded {len(df)} samples")
            logger.info(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def preprocess_texts(self, df):
        """
        Apply text preprocessing to all texts

        Why: Clean and normalize text for better feature extraction
        """
        logger.info("Preprocessing texts...")

        text_col = self.config['data']['text_column']

        # Apply preprocessing
        df['text_clean'] = df[text_col].apply(
            lambda x: preprocess_text(x, self.config)
        )

        # Remove empty texts
        min_length = self.config['preprocessing']['min_text_length']
        initial_len = len(df)
        df = df[df['text_clean'].str.len() >= min_length].copy()
        removed = initial_len - len(df)

        if removed > 0:
            logger.info(f"  Removed {removed} texts below minimum length")

        logger.info(f"✓ Preprocessed {len(df)} texts")
        return df

    def extract_labels(self, df):
        """
        Extract emotion labels from labels_str column

        Why: Convert string representation to clean labels
        """
        logger.info("Extracting emotion labels...")

        label_col = self.config['data']['label_column']
        df['emotion'] = df[label_col].apply(extract_emotion_label)

        # Check label distribution
        label_dist = df['emotion'].value_counts()
        logger.info(f"\n  Label distribution:\n{label_dist}")

        return df

    def create_splits(self, df):
        """
        Create stratified train/val/test splits

        Why stratified: Maintain emotion distribution across splits
        Critical for imbalanced emotion classes
        """
        logger.info("Creating train/val/test splits...")

        train_ratio = self.config['data']['train_split']
        val_ratio = self.config['data']['val_split']
        test_ratio = self.config['data']['test_split']
        seed = self.config['data']['random_seed']

        # First split: train + (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df['emotion'],
            random_state=seed
        )

        # Second split: val + test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['emotion'],
            random_state=seed
        )

        logger.info(f"✓ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        return train_df, val_df, test_df

    def vectorize_texts(self, train_df, val_df, test_df):
        """
        Convert text to TF-IDF features

        Why TF-IDF:
        - Converts text to numerical features
        - Weights important terms higher
        - Works well with scikit-learn models
        """
        logger.info("Generating TF-IDF features...")

        max_features = self.config['preprocessing']['tfidf_max_features']
        ngram_range = tuple(self.config['preprocessing']['tfidf_ngram_range'])

        # Initialize vectorizer
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )

        # Fit on training data only
        X_train = tfidf.fit_transform(train_df['text_clean'])
        X_val = tfidf.transform(val_df['text_clean'])
        X_test = tfidf.transform(test_df['text_clean'])

        logger.info(f"✓ TF-IDF shape: {X_train.shape}")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Vocabulary size: {len(tfidf.vocabulary_)}")

        return X_train, X_val, X_test, tfidf

    def save_processed_data(self, X_train, X_val, X_test, 
                           y_train, y_val, y_test,
                           tfidf_vectorizer,
                           train_df, val_df, test_df):
        """Save all processed data"""
        logger.info("Saving preprocessed data...")

        # Save sparse matrices (TF-IDF features)
        save_npz(self.output_dir / 'X_train.npz', X_train)
        save_npz(self.output_dir / 'X_val.npz', X_val)
        save_npz(self.output_dir / 'X_test.npz', X_test)

        # Save labels
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_val.npy', y_val)
        np.save(self.output_dir / 'y_test.npy', y_test)

        # Save vectorizer (needed for inference)
        with open(self.output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)

        # Save raw dataframes (for reference)
        train_df.to_csv(self.output_dir / 'train_raw.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_raw.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_raw.csv', index=False)

        # Save emotion labels list
        emotions = sorted(y_train.unique())
        with open(self.output_dir / 'emotions.pkl', 'wb') as f:
            pickle.dump(emotions, f)

        logger.info(f"✓ All files saved to {self.output_dir}")

    def run(self):
        """Execute complete preprocessing pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*80 + "\n")

        # 1. Load data
        df = self.load_data()

        # 2. Preprocess texts
        df = self.preprocess_texts(df)

        # 3. Extract labels
        df = self.extract_labels(df)

        # 4. Create splits
        train_df, val_df, test_df = self.create_splits(df)

        # 5. Vectorize
        X_train, X_val, X_test, tfidf = self.vectorize_texts(
            train_df, val_df, test_df
        )

        # 6. Extract labels
        y_train = train_df['emotion'].values
        y_val = val_df['emotion'].values
        y_test = test_df['emotion'].values

        # 7. Save everything
        self.save_processed_data(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            tfidf,
            train_df, val_df, test_df
        )

        logger.info("\n" + "="*80)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info("="*80 + "\n")

        return {
            'n_train': len(train_df),
            'n_val': len(val_df),
            'n_test': len(test_df),
            'n_features': X_train.shape[1]
        }

if __name__ == '__main__':
    preprocessor = LocalPreprocessor()
    results = preprocessor.run()

    print("\n✓ Preprocessing Summary:")
    print(f"  Training samples: {results['n_train']}")
    print(f"  Validation samples: {results['n_val']}")
    print(f"  Test samples: {results['n_test']}")
    print(f"  Features: {results['n_features']}")
