# EMOTION ANALYSIS - LOCAL TESTING
================================================================================

## 🎯 QUICK START (5 Minutes)

### Prerequisites
- Python 3.8 or higher
- 500 MB free disk space
- Internet connection (for dependency installation)

### Installation

```bash
# 1. Create project directory
mkdir emotion_analysis_local
cd emotion_analysis_local

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
# Mac/Linux:
source venv/bin/activate

# Windows PowerShell:
.\venv\Scripts\Activate

# Windows Command Prompt:
venv\Scripts\activate.bat

# 4. Install dependencies
pip install -r requirements_local.txt

# 5. Place Emotion_Lexicon.csv in data/ folder
mkdir data
# Copy your Emotion_Lexicon.csv to data/

# 6. Run complete pipeline
python 07_run_all_tests.py
```

**That's it!** The pipeline will:
1. ✅ Check dependencies
2. ✅ Generate 1000 test samples
3. ✅ Preprocess data
4. ✅ Train models with nested CV
5. ✅ Run inference demo

**Expected time:** 10-15 minutes on CPU

---

## 📁 PROJECT STRUCTURE

```
emotion_analysis_local/
├── 01_config_local.yaml           # Configuration file
├── 02_utils_local.py               # Utility functions
├── 03_preprocess_local.py          # Data preprocessing
├── 04_train_local.py               # Model training (nested CV)
├── 05_inference_local.py           # Inference engine
├── 06_create_test_data.py          # Test data generator
├── 07_run_all_tests.py             # Complete pipeline
├── requirements_local.txt          # Python dependencies
├── README_LOCAL_TESTING.md         # This file
│
├── data/
│   ├── Emotion_Lexicon.csv         # NRC emotion lexicon (you provide)
│   └── test_sample.csv             # Generated test data (1000 samples)
│
├── outputs/
│   ├── preprocessed_batches/       # Processed data
│   │   ├── X_train.npz             # Training features
│   │   ├── y_train.npy             # Training labels
│   │   ├── tfidf_vectorizer.pkl    # TF-IDF model
│   │   └── ...
│   │
│   ├── trained_models/             # Trained models
│   │   ├── lexicon_model.pkl       # Lexicon-based model
│   │   ├── nb_model.pkl            # Naive Bayes
│   │   ├── svm_model.pkl           # SVM
│   │   └── ...
│   │
│   └── cv_results/                 # Cross-validation results
│       ├── cv_results.csv          # Fold-by-fold metrics
│       └── cv_results.json         # Summary statistics
│
└── logs/
    └── training_log.txt            # Detailed execution logs
```

---

## 🚀 USAGE EXAMPLES

### Example 1: Run Complete Pipeline

```bash
python 07_run_all_tests.py
```

**Output:**
```
================================================================================
            EMOTION ANALYSIS - LOCAL TESTING PIPELINE
================================================================================

================================================================================
                     STEP 1/5: CHECKING DEPENDENCIES
================================================================================

✓ All core dependencies installed
✓ NLTK data available

================================================================================
                      STEP 2/5: CREATING TEST DATA
================================================================================

✓ Test data already exists (data/test_sample.csv)

...

✓ All steps completed successfully!
⏱ Total time: 12m 34s
```

### Example 2: Individual Steps

```bash
# Step 1: Create test data
python 06_create_test_data.py

# Step 2: Preprocess
python 03_preprocess_local.py

# Step 3: Train models
python 04_train_local.py

# Step 4: Run inference
python 05_inference_local.py
```

### Example 3: Custom Inference

```python
from inference_local import EmotionInference

# Load models
inference = EmotionInference()

# Single prediction
result = inference.predict("I am so happy today!")
print(result['predictions'])  # Output: 'Joy'

# Batch prediction
texts = [
    "This is amazing!",
    "I am very angry",
    "This is scary"
]
results = inference.predict(texts)
print(results['predictions'])
# Output: ['Joy', 'Anger', 'Fear']

# Detailed prediction with explanation
inference.predict_with_explanation("I am excited about tomorrow!")
```

**Output:**
```
================================================================================
TEXT: I am excited about tomorrow!
================================================================================

Cleaned: excit tomorrow

PREDICTIONS:
  Lexicon:     Anticipation
  Naive Bayes: Joy
  SVM:         Anticipation
  ─────────────
  Ensemble:    Anticipation

Confidence: 0.847

Probability Distribution:
  Anger          : 0.023
  Anticipation   : 0.847
  Disgust        : 0.012
  Fear           : 0.034
  Joy            : 0.042
  Sadness        : 0.018
  Surprise       : 0.015
  Trust          : 0.009
================================================================================
```

---

## ⚙️ CONFIGURATION

Edit `01_config_local.yaml` to customize:

```yaml
data:
  input_file: "data/test_sample.csv"  # Your data file
  batch_size: 500                      # Batch size

preprocessing:
  tfidf_max_features: 1000             # Number of features

cross_validation:
  outer_folds: 2                       # Outer CV folds
  inner_folds: 2                       # Inner CV folds

models:
  naive_bayes:
    enabled: true
  svm:
    enabled: true
    kernel: "linear"  # "linear" faster on CPU than "rbf"
```

---

## 📊 EXPECTED RESULTS

### Test Dataset (1000 samples)

**Preprocessing:**
- Train: 700 samples
- Validation: 100 samples
- Test: 200 samples
- Features: ~1000 TF-IDF features

**Training Time:** 10-15 minutes on CPU

**Expected Performance (F1 Macro):**
- Lexicon: 0.35-0.45
- Naive Bayes: 0.40-0.50
- SVM: 0.45-0.55
- Ensemble: 0.48-0.58

**Note:** Performance on 1000 samples is lower than production. With 500k samples on AWS GPU, expect F1 = 0.65-0.75

---

## 🔧 TROUBLESHOOTING

### Problem: "ModuleNotFoundError"

```bash
# Solution: Activate virtual environment
source venv/bin/activate  # Mac/Linux
.\venv\Scripts\Activate  # Windows

# Reinstall dependencies
pip install -r requirements_local.txt
```

### Problem: "FileNotFoundError: data/Emotion_Lexicon.csv"

```bash
# Solution: Place lexicon file in data/ folder
mkdir data
# Copy Emotion_Lexicon.csv to data/
```

### Problem: "NLTK data not found"

```python
# Solution: Download NLTK data manually
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

### Problem: Training too slow (>30 minutes)

```yaml
# Solution: Reduce folds in 01_config_local.yaml
cross_validation:
  outer_folds: 2  # Already minimal
  inner_folds: 2  # Already minimal

# Or use smaller test dataset
# Edit 06_create_test_data.py:
create_test_dataset(n_samples=500)  # Instead of 1000
```

### Problem: "Memory Error"

```yaml
# Solution: Reduce features in config
preprocessing:
  tfidf_max_features: 500  # Instead of 1000

# Or reduce batch size
data:
  batch_size: 250  # Instead of 500
```

---

## 📈 NEXT STEPS

### After Local Testing Succeeds:

1. **Review Results**
   ```bash
   # Check CV metrics
   cat outputs/cv_results/cv_results.json

   # Check detailed logs
   cat logs/training_log.txt
   ```

2. **Test Your Own Data**
   - Replace `data/test_sample.csv` with your data
   - Ensure columns: `text`, `labels_str`
   - Run: `python 03_preprocess_local.py`

3. **Deploy to AWS GPU**
   - Update config for GPU: `use_gpu: true`
   - Change dataset to full 500k samples
   - Follow `AWS_SETUP_GUIDE.md`
   - Expected F1 improvement: 0.50 → 0.70

4. **Optimize Models**
   - Tune hyperparameters in config
   - Try different kernels: `rbf` instead of `linear`
   - Increase features: `tfidf_max_features: 5000`

---

## 🎓 UNDERSTANDING THE WORKFLOW

### Why Nested Cross-Validation?

**Standard CV (WRONG):**
```
1. Tune hyperparameters on full data
2. Evaluate on same data
→ Result: Overfitted to data!
```

**Nested CV (CORRECT):**
```
OUTER LOOP (evaluation):
  Hold out test set
  INNER LOOP (tuning):
    Try different hyperparameters
    Select best on validation set
  Train with best config
  Evaluate on held-out test set
→ Result: Unbiased performance!
```

**Our Implementation:**
- Outer folds: 2 (model evaluation)
- Inner folds: 2 (hyperparameter selection)
- Stratified: Maintains emotion distribution
- Total models trained: 2×2 = 4 per algorithm

### Why Ensemble?

Individual model weaknesses:
- **Lexicon:** Rule-based, misses context
- **Naive Bayes:** Assumes independence (not always true)
- **SVM:** Can overfit

**Ensemble (Majority Voting):**
- Combines strengths
- Reduces individual weaknesses
- Usually 5-10% better F1 than best single model

### Model Comparison

| Model | Training Time | Interpretability | Accuracy |
|-------|---------------|------------------|----------|
| Lexicon | Instant | ⭐⭐⭐⭐⭐ | Medium |
| Naive Bayes | Fast (~10s) | ⭐⭐⭐⭐ | Good |
| SVM | Medium (~2min) | ⭐⭐ | Best |
| Ensemble | All above | ⭐⭐⭐ | Best+ |

---

## 💡 TIPS & BEST PRACTICES

### Development Tips
1. **Always test locally first** - Catches bugs for free
2. **Use small datasets** - 1000 samples enough for testing
3. **Check logs** - `logs/training_log.txt` has detailed info
4. **Version control** - Git commit after each successful run

### Performance Tips
1. **CPU optimization:**
   - Use `kernel="linear"` for SVM (faster than `rbf`)
   - Reduce `tfidf_max_features` (fewer dimensions)
   - Reduce CV folds (2×2 instead of 5×3)

2. **When to use GPU:**
   - Dataset > 10,000 samples
   - Need deep learning (LSTM, BERT)
   - Time-sensitive (want results in hours, not days)

### Production Readiness
```python
# Before deploying to production:

# 1. Evaluate on held-out test set
test_results = evaluate_on_test_set()
assert test_results['f1_macro'] > 0.50, "Performance too low"

# 2. Test edge cases
edge_cases = ["", "!!!!", "not not not good"]
predictions = inference.predict(edge_cases)
# Verify no crashes

# 3. Test performance
import time
start = time.time()
inference.predict(["test"] * 1000)
elapsed = time.time() - start
print(f"Throughput: {1000/elapsed:.0f} predictions/sec")
```

---

## 📞 SUPPORT & RESOURCES

### Documentation
- Configuration: See `01_config_local.yaml` comments
- API reference: Docstrings in each `.py` file
- AWS deployment: See `AWS_SETUP_GUIDE.md`

### Common Questions

**Q: Can I use my own dataset?**
A: Yes! Format as CSV with `text` and `labels_str` columns.

**Q: How to improve accuracy?**
A: 1) More training data, 2) Tune hyperparameters, 3) Add features

**Q: How long does training take?**
A: 10-15 min for 1k samples on CPU, 2-4 hours for 500k on GPU

**Q: Can I add more emotions?**
A: Yes! Update `emotions.names` in config and retrain.

**Q: How to export models?**
A: Models auto-saved as `.pkl` in `outputs/trained_models/`

---

## ✅ SUCCESS CHECKLIST

Before moving to AWS:

- [ ] All dependencies installed
- [ ] Virtual environment activated
- [ ] Emotion_Lexicon.csv in data/ folder
- [ ] Test data created (1000 samples)
- [ ] Preprocessing completed successfully
- [ ] Training completed (2 outer folds)
- [ ] Inference produces predictions
- [ ] No errors in logs
- [ ] F1 score > 0.35 (reasonable for small data)

**If all checked:** ✅ Ready for AWS deployment!

---

## 📝 LICENSE & CITATION

This code is provided for educational and research purposes.

**Lexicon Citation:**
```
Mohammad, S., & Turney, P. (2013). 
"Crowdsourcing a Word-Emotion Association Lexicon."
Computational Intelligence, 29(3), 436-465.
```

---

## 🙏 ACKNOWLEDGMENTS

- NRC Word-Emotion Association Lexicon
- scikit-learn library
- NLTK toolkit
- Python community

---

**Need help?** Check troubleshooting section or open an issue!

**Ready for production?** Follow AWS_SETUP_GUIDE.md for GPU training!
