# Multi-Model Late-Fusion City Classifier

Predicts which of 4 cities (Dubai, New York City, Paris, Rio de Janeiro) a survey response
describes, using a late-fusion ensemble of 5 independently-trained sub-models.

**90.8% held-out test accuracy** (25% random baseline), validated as **89.5% mean ± 0.7pp**
across 5 random data splits.

Full write-up: [`CSC311 ML Challenge Final Report.pdf`](./CSC311%20ML%20Challenge%20Final%20Report.pdf)

## Approach

Rather than concatenating all survey questions into one feature vector ("early fusion"), each
question is modeled independently and the resulting class-probability vectors are combined
("late fusion"). This let each sub-model specialize on the structure of its own question
(e.g. polynomial interactions for numeric ratings, a decision tree for categorical keyword
indicators) rather than forcing one model to find a single decision boundary across
heterogeneous feature types.

| Sub-model | Question | Model | Test accuracy |
|---|---|---|---|
| Q1–4 | 1–5 ratings (popularity, virality, architecture, street-party enthusiasm) | Softmax logistic regression (hand-implemented) | 63.5% |
| Q5 | Travel companion (multi-select) | Softmax logistic regression | 41.0% |
| Q6 | Category relatability ranking | Softmax logistic regression | 73.7% |
| Q7 | Guessed January temperature | Softmax logistic regression | 60.4% |
| Q10 | Free-text quote (keyword-engineered) | Decision tree (hand-reimplemented in NumPy) | 56.0% |
| **Ensemble** | **All of the above, uniformly averaged** | **Late fusion** | **90.8%** |

Softmax regression (linear layer → softmax → cross-entropy loss → gradient descent) and the
fitted decision tree's split logic are both implemented from scratch in NumPy for the final
prediction path — scikit-learn is used only during exploration (see below).

Four model families were benchmarked in total (logistic regression, decision trees, KNN,
Gaussian Naive Bayes) against 2 fusion strategies (early vs. late), with results and reasoning
in the report.

## Repo structure

```
prediction code/
  pred.py                 # final inference script — loads ensemble_weights.npz, predicts city from raw CSV
  ensemble_weights.npz    # trained parameters (softmax weights, tree arrays, standardization stats)

development and experimentation/
  Q1-Q4.ipynb, Q5.ipynb, Q6.ipynb, Q7.ipynb, Q10.ipynb   # per-question model exploration & tuning
  Q8.ipynb, Q9.ipynb      # exploratory only — excluded from final model (see report, Section 1.2)
  feature_lib.py          # shared feature engineering (polynomial expansion, standardization, keyword extraction)
  train_ensemble.py       # reproduces the fixed data split, trains all sub-models, saves ensemble_weights.npz
  extra_models_exploration.ipynb   # KNN / Naive Bayes / early-fusion baselines

cleaned_dataset.csv
CSC311 ML Challenge Final Report.pdf
```

## Running predictions

```bash
cd "prediction code"
python pred.py path/to/test.csv
```

Loads all learned parameters from `ensemble_weights.npz` (no retraining required) and outputs
a predicted city per row using only NumPy and pandas at runtime.

## Reproducing training

```bash
cd "development and experimentation"
python train_ensemble.py
```

Regenerates the fixed stratified train/val/test split (seed=42), retrains each sub-model, and
re-saves `ensemble_weights.npz`.

## Key design decisions

- **No data leakage:** all imputation medians/means and standardization statistics are fit on
  the training split only, and reused as-is on validation/test data.
- **Q8 and Q9 excluded:** both are open-ended numeric estimates with implausible outliers
  (max values up to 10,000) and weak decision-tree validation accuracy (~35–39%) — see
  Section 1.2 of the report for the full justification.
- **Uniform averaging over learned stacking:** a learned combiner was tested but overfit on
  the small (~150-row) fusion-tuning set; simple averaging was more robust across seeds
  (Section 3.4 of the report).

## Stack

Python, NumPy, pandas, scikit-learn (exploration only), Matplotlib
