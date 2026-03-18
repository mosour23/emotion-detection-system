# Test Samples for Emotion Detection

This directory contains sample test files for batch evaluation and demonstration of the emotion detection model.

## Overview

Each file contains 10 representative sentences for a specific emotion class. These samples can be used to:

1. **Demonstrate the model in action** without training
2. **Validate model predictions** on hand-crafted examples
3. **Test batch processing** with the `batch_test.py` script
4. **Verify model sensitivity** to emotion-specific language patterns

## Files

### `sadness_samples.txt`
10 sentences expressing sadness, despair, and melancholy.
Expected emotion: **sadness** 😢

### `joy_samples.txt`
10 sentences expressing happiness, excitement, and elation.
Expected emotion: **joy** 😊

### `love_samples.txt`
10 sentences expressing love, affection, and deep affection.
Expected emotion: **love** ❤️

### `anger_samples.txt`
10 sentences expressing anger, rage, and frustration.
Expected emotion: **anger** 😠

### `fear_samples.txt`
10 sentences expressing fear, terror, and anxiety.
Expected emotion: **fear** 😨

### `surprise_samples.txt`
10 sentences expressing surprise, amazement, and shock.
Expected emotion: **surprise** 😲

## Usage

### Run Batch Tests

Test all emotion classes:
```bash
python batch_test.py
```

Test a specific emotion:
```bash
python batch_test.py --emotion sadness
```

Show prediction probabilities:
```bash
python batch_test.py --show-proba
```

Test specific emotion with probabilities:
```bash
python batch_test.py --emotion joy --show-proba
```

### Example Output

```
======================================================================
BATCH TEST: SADNESS 😢
======================================================================
Accuracy on sadness: 100.0% (10/10)

✓ [1] I feel completely devastated after losing my job.
    → Predicted: SADNESS 😢

✓ [2] Everything feels hopeless and I cannot stop crying.
    → Predicted: SADNESS 😢
    
...
======================================================================
Batch testing complete!
```

## Adding Custom Test Files

To add your own test files:

1. Create a `.txt` file in this directory
2. Name it `{emotion}_samples.txt` where `{emotion}` is one of:
   - `sadness`
   - `joy`
   - `love`
   - `anger`
   - `fear`
   - `surprise`

3. Add one sentence per line
4. Run batch tests to evaluate

Example:
```bash
echo "I am absolutely thrilled about this opportunity!" >> custom_joy.txt
python batch_test.py --emotion joy
```

## Requirements

- Trained model artifacts in `models/saved/`:
  - `svm_classifier.pkl`
  - `tfidf_extractor.pkl`
  - `label_encoder.pkl`

If these files don't exist, run the training pipeline first:
```bash
python train_pipeline.py
```

## Notes

- Samples are designed to be clear and representative
- Original dataset uses synthetic balanced samples (800 per class)
- These test files provide quick validation without retraining
- For comprehensive evaluation, see the test set results in the training pipeline output

## Citation

If using these test samples in research or publication, please cite:
> Emotion Detection System - APU Pattern Recognition CT104-3-M
