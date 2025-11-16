"""
EMOTION ANALYSIS - LOCAL UTILITIES MODULE
==========================================
Purpose: Core utility functions for local testing
Author: Data Science Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import re
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(log_file='logs/training_log.txt', verbose=True):
    """
    Initialize logging configuration

    Args:
        log_file: Path to log file
        verbose: Print to console

    Returns:
        logging.Logger instance
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('emotion_analysis_local')
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Initialize logger
logger = setup_logger()

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_file='01_config_local.yaml'):
    """Load YAML configuration file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_text(text, config=None):
    """
    Comprehensive text preprocessing pipeline

    Steps:
    1. Lowercase conversion
    2. URL removal
    3. Mention removal (@username)
    4. Negation handling (not good → not_good)
    5. Special character removal
    6. Stopword removal
    7. Stemming

    Args:
        text: Input text string
        config: Configuration dict (optional)

    Returns:
        Cleaned text string
    """
    # Import here to avoid loading if not needed
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    if pd.isna(text) or text == '':
        return ''

    text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove mentions
    text = re.sub(r'@\w+', '', text)

    # 4. Handle negations (preserve semantic meaning)
    negation_words = [
        'not', 'no', 'never', 'neither', 'nobody', 'nothing',
        'nowhere', "can't", "cannot", "couldn't", "doesn't",
        "don't", "hadn't", "hasn't", "haven't", "isn't",
        "shouldn't", "wasn't", "weren't", "won't", "wouldn't"
    ]

    for neg_word in negation_words:
        # Replace "not good" with "not_good"
        pattern = f'\\b{neg_word}\\s+(\\w+)'
        text = re.sub(pattern, f'not_\\1', text)

    # 5. Remove special characters
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)

    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Tokenization
    tokens = text.split()

    # 8. Stopword removal (keep negated words)
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words or t.startswith('not_')]
    except LookupError:
        logger.warning("NLTK stopwords not found, skipping stopword removal")

    # 9. Stemming
    try:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    except:
        logger.warning("Stemming failed, using original tokens")

    # 10. Rejoin
    text = ' '.join(tokens)

    return text if text else ''

# ============================================================================
# LEXICON HANDLING
# ============================================================================

def load_emotion_lexicon(lexicon_file):
    """
    Load NRC emotion lexicon

    Returns:
        tuple: (lexicon_dict, emotion_columns)
    """
    logger.info(f"Loading emotion lexicon from {lexicon_file}")

    try:
        lexicon_df = pd.read_csv(lexicon_file)

        # Get emotion columns (all except first)
        emotion_cols = [col for col in lexicon_df.columns if col != 'English.Word']

        # Convert to dictionary for fast lookup
        lexicon_dict = {}
        for idx, row in lexicon_df.iterrows():
            word = str(row['English.Word']).lower()
            emotions = {col: int(row[col]) for col in emotion_cols}
            lexicon_dict[word] = emotions

        logger.info(f"✓ Loaded {len(lexicon_dict)} words")
        logger.info(f"✓ Emotions: {emotion_cols}")

        return lexicon_dict, emotion_cols

    except Exception as e:
        logger.error(f"Failed to load lexicon: {e}")
        raise

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(y_true, y_pred, model_name='Model'):
    """
    Calculate comprehensive classification metrics

    Metrics:
    - Accuracy
    - Precision (macro & weighted)
    - Recall (macro & weighted)
    - F1-score (macro & weighted)
    - Confusion matrix
    - Per-class metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model identifier

    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # Add confusion matrix
    try:
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    except:
        logger.warning("Could not generate confusion matrix")

    return metrics

def print_metrics(metrics):
    """Pretty print metrics"""
    print(f"\n{'='*80}")
    print(f"MODEL: {metrics['model_name'].upper()}")
    print(f"{'='*80}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro):  {metrics['f1_macro']:.4f}")
    print(f"\nWeighted Metrics:")
    print(f"Precision:         {metrics['precision_weighted']:.4f}")
    print(f"Recall:            {metrics['recall_weighted']:.4f}")
    print(f"F1-Score:          {metrics['f1_weighted']:.4f}")
    print(f"{'='*80}\n")

# ============================================================================
# MODEL SERIALIZATION
# ============================================================================

def save_model(model, model_path, format='pkl'):
    """
    Save model to disk

    Args:
        model: Model object
        model_path: Path to save (without extension)
        format: 'pkl' or 'joblib'
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if format == 'pkl':
        with open(f"{model_path}.pkl", 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✓ Saved model to {model_path}.pkl")

    elif format == 'joblib':
        import joblib
        joblib.dump(model, f"{model_path}.joblib")
        logger.info(f"✓ Saved model to {model_path}.joblib")

def load_model(model_path):
    """Load model from disk"""
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_path.endswith('.joblib'):
        import joblib
        return joblib.load(model_path)

# ============================================================================
# DATA UTILITIES
# ============================================================================

def extract_emotion_label(label_str):
    """
    Extract single emotion from label string

    Example: "['Joy']" → "Joy"
    """
    import ast
    try:
        labels_list = ast.literal_eval(str(label_str))
        if isinstance(labels_list, list) and len(labels_list) > 0:
            return labels_list[0]
    except:
        pass
    return 'Neutral'

def create_batches(data, batch_size):
    """
    Split data into batches

    Args:
        data: DataFrame
        batch_size: Size per batch

    Yields:
        DataFrame batches
    """
    n_batches = int(np.ceil(len(data) / batch_size))

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        yield data.iloc[start_idx:end_idx].copy()

# ============================================================================
# REPORTING
# ============================================================================

def save_cv_results(cv_results, output_dir='outputs/cv_results'):
    """Save cross-validation results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save as CSV
    if isinstance(cv_results, pd.DataFrame):
        cv_results.to_csv(f"{output_dir}/cv_results.csv", index=False)
    elif isinstance(cv_results, dict):
        import json
        with open(f"{output_dir}/cv_results.json", 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)

    logger.info(f"✓ Saved CV results to {output_dir}")

# ============================================================================
# DOWNLOAD NLTK DATA (RUN ONCE)
# ============================================================================

def download_nltk_data():
    """Download required NLTK data"""
    import nltk

    packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']

    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)

# Run on import
try:
    download_nltk_data()
except:
    logger.warning("Could not download NLTK data automatically")
