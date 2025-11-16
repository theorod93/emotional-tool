# Emotion Analysis Pipeline (SuperEmotion)

End‑to‑end emotion analysis pipeline for social listening and marketing insights.  
The goal is to train classical ML and lightweight DL models (no heavy LLMs) on a large emotion dataset (≈500k samples), with:

- Batch processing for scalable training.
- Nested cross‑validation for robust evaluation.
- Lexicon‑based baseline using the NRC Emotion Lexicon.
- CPU‑friendly local testing and optional GPU acceleration on AWS.

---

## Features

- **Data preprocessing**
  - Cleans text, handles negation, removes noise.
  - Stratified train/validation/test splits (70/10/20).
  - TF‑IDF feature extraction; configurable n‑grams and vocabulary size.

- **Emotion modelling**
  - Lexicon‑based classifier (NRC Emotion Lexicon).
  - Naive Bayes and SVM models.
  - (Optional) Lightweight neural models for GPU runs.
  - Support for 8 emotions (Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust).

- **Evaluation**
  - Nested cross‑validation (outer folds for evaluation, inner folds for tuning).
  - Macro and weighted Precision / Recall / F1.
  - Confusion matrices and fold‑by‑fold logging.

- **Deployment‑ready artifacts**
  - Saved models (`.pkl`) and TF‑IDF vectorizer.
  - Inference script for local batch or single‑text predictions.

---

## Project Structure

.emotion-analysis/
├── src/
│ ├── init.py
│ ├── config/
│ │ └── config_local.yaml
│ ├── data/
│ │ ├── init.py
│ │ ├── preprocess_local.py
│ │ └── create_test_data.py
│ ├── models/
│ │ ├── init.py
│ │ └── train_local.py
│ ├── inference/
│ │ ├── init.py
│ │ └── inference_local.py
│ └── utils/
│ ├── init.py
│ └── utils_local.py
│
├── data/
│ ├── Emotion_Lexicon.csv # Not tracked (see .gitignore), add locally
│ └── super_emotion_dataset.csv # Not tracked (large / private)
│
├── outputs/
│ ├── preprocessed_batches/
│ ├── trained_models/
│ └── cv_results/
│
├── notebooks/
│ └── exploratory.ipynb
│
├── tests/
│ ├── init.py
│ ├── test_utils.py
│ └── test_inference.py
│
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── Makefile
└── pyproject.toml (or setup.py)

text

---

## Installation

### 1. Clone the repository

git clone https://github.com/<your-username>/emotion-analysis.git
cd emotion-analysis

text

### 2. Create and activate a virtual environment

python3 -m venv .venv

macOS / Linux
source .venv/bin/activate

Windows (PowerShell)
..venv\Scripts\Activate

text

### 3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

(optional) dev tools: linting, formatting, testing
pip install -r requirements-dev.txt

text

---

## Data Setup

1. Create the `data/` directory if it does not exist:

mkdir -p data

text

2. Add the following files locally (they are intentionally **not** committed):

- `data/Emotion_Lexicon.csv`  
  NRC Emotion Lexicon (respect its license; do not commit).

- `data/super_emotion_dataset.csv`  
  Your 500k‑sample dataset with columns:
  - `text`
  - `labels` (numeric emotion IDs)
  - `labels_str` (string labels)
  - `labels_source` (original source labels; can differ)

> Only `labels` and `labels_str` are used as ground truth; `labels_source` is ignored in training.

---

## Local Usage

### 1. Generate a small test dataset (optional, for dry runs)

python -m src.data.create_test_data

text

This creates `data/test_sample.csv` with ≈1000 synthetic samples.

### 2. Preprocess data

Configure `src/config/config_local.yaml` to point to either:

- `data/test_sample.csv` for local testing, or
- `data/super_emotion_dataset.csv` for the real dataset.

Then run:

python -m src.data.preprocess_local

text

This will:

- Clean and tokenize text.
- Create stratified train/val/test splits (70/10/20).
- Fit a TF‑IDF vectorizer on the training set.
- Save features, labels, and vectorizer under `outputs/preprocessed_batches/`.

### 3. Train models with nested cross‑validation

python -m src.models.train_local

text

This will:

- Run nested CV (2×2 for local testing; higher for production).
- Train:
  - Lexicon baseline
  - Naive Bayes
  - SVM
- Save trained models to `outputs/trained_models/`.
- Save CV metrics to `outputs/cv_results/`.

### 4. Run inference

python -m src.inference.inference_local

text

This script:

- Loads the TF‑IDF vectorizer and trained models.
- Prints predictions for a few example texts.
- Can be imported as a module to integrate into other systems.

---

## Configuration

All main settings live in `src/config/config_local.yaml`:

- Data paths
- Batch size
- TF‑IDF parameters
- Cross‑validation folds
- Model hyperparameters
- Hardware options (CPU/GPU flag)

Example snippet:

data:
input_file: "data/test_sample.csv"
text_column: "text"
label_column: "labels_str"
batch_size: 500
train_split: 0.70
val_split: 0.10
test_split: 0.20

preprocessing:
tfidf_max_features: 1000
tfidf_ngram_range:​
min_text_length: 3

cross_validation:
outer_folds: 2
inner_folds: 2

hardware:
use_gpu: false
device: "cpu"

text

---

## Development

### Tests

pytest -q

text

### Linting & Formatting

ruff check src tests
black src tests

text

Use the `Makefile` to bundle common tasks:

make test
make lint
make format
make all

text

---

## Environment Variables

Sensitive configuration (e.g. AWS keys) should **never** be committed.

Use `.env.example` as a template:

cp .env.example .env

edit .env with your secrets
text

Add your real values in `.env` (which is ignored by git).

---

## License

By default this repo uses the MIT License (see `LICENSE`), but you can change it if needed.

Make sure to respect third‑party licenses (e.g. NRC Emotion Lexicon).

---

## Roadmap

- [ ] Add GPU‑accelerated training script for AWS EC2 / Docker.
- [ ] Add LSTM / lightweight CNN model for sequence modeling.
- [ ] Add CLI interface for batch inference.
- [ ] Add Dockerfile and GitHub Actions CI.

---

## Citation

If you use the NRC Emotion Lexicon, please cite:

> Mohammad, S., & Turney, P. (2013).  
> *Crowdsourcing a Word–Emotion Association Lexicon.*  
> Computational Intelligence, 29(3), 436‑465.
