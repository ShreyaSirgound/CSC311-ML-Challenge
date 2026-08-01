import re
import numpy as np
import pandas as pd

# Canonical orderings (must be fixed so train-time and test-time feature
# matrices always line up column-for-column)
Q5_CATEGORIES = ['Co-worker', 'Friends', 'Partner', 'Siblings', 'Unknown']
Q6_CATEGORIES = ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']

CITY_KEYWORDS = {
    'Dubai': ['rich', 'money', 'come', 'habibi', 'world', 'burj', 'oil', 'tallest', 'khalifa', 'desert'],
    'Paris': ['love', 'tower', 'oui', 'eiffel', 'baguette', 'romance', 'romantic', 'always', 'fashion'],
    'New York City': ['dreams', 'made', 'concrete', 'jungle', 'never', 'sleeps', 'im', 'big', 'apple', 'here'],
    'Rio de Janeiro': ['football', 'life', 'brazil', 'de', 'carnival', 'soccer', 'beautiful', 'party', 'samba', 'jesus'],
}
Q10_FEATURE_NAMES = []
for _kws in CITY_KEYWORDS.values():
    for _kw in _kws:
        if _kw not in Q10_FEATURE_NAMES:
            Q10_FEATURE_NAMES.append(_kw)


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# Q1-Q4: numeric ratings -> degree-2 polynomial feature block
def _raw_q1_4_poly(df, fill_values):
    q1 = df['Q1'].fillna(fill_values['Q1']).to_numpy(dtype=float)
    q2 = df['Q2'].fillna(fill_values['Q2']).to_numpy(dtype=float)
    q3 = df['Q3'].fillna(fill_values['Q3']).to_numpy(dtype=float)
    q4 = df['Q4'].fillna(fill_values['Q4']).to_numpy(dtype=float)
    return np.column_stack([
        q1, q2, q3, q4,
        q1**2, q2**2, q3**2, q4**2,
        q1*q2, q1*q3, q1*q4,
        q2*q3, q2*q4, q3*q4,
    ])


def build_q1_4_features(df, fill_values, mu, sigma):
    poly = _raw_q1_4_poly(df, fill_values)
    poly_std = (poly - mu) / sigma
    return add_bias(poly_std)


def fit_q1_4_stats(df, train_idx):
    train = df.iloc[train_idx]
    fill_values = {q: train[q].median() for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    train_poly = _raw_q1_4_poly(train, fill_values)
    mu = train_poly.mean(axis=0)
    sigma = train_poly.std(axis=0)
    sigma[sigma == 0] = 1.0
    return fill_values, mu, sigma


# Q5: companions -> multi-hot indicator block (fixed column order)
def build_q5_features(df):
    q5 = df['Q5'].fillna('Unknown').astype(str)
    q5_clean = q5.str.replace(r'\s*,\s*', ',', regex=True)
    dummies = q5_clean.str.get_dummies(sep=',')
    # Force the fixed column order, any category never seen becomes all-zero
    for cat in Q5_CATEGORIES:
        if cat not in dummies.columns:
            dummies[cat] = 0
    dummies = dummies[Q5_CATEGORIES]
    return add_bias(dummies.to_numpy(dtype=float))


# Q6: ranking string -> 6 numeric ranks, NaNs filled with TRAIN column means
def _parse_q6_string(q6_str):
    row = {}
    if not isinstance(q6_str, str):
        return row
    for pair in q6_str.split(','):
        pair = pair.strip()
        if '=>' in pair:
            k, v = pair.split('=>', 1)
            k, v = k.strip(), v.strip()
            if k and v:
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
    return row


def _raw_q6_matrix(df, fill_means):
    parsed = [_parse_q6_string(v) for v in df['Q6']]
    q6_df = pd.DataFrame(parsed, index=df.index)
    for cat in Q6_CATEGORIES:
        if cat not in q6_df.columns:
            q6_df[cat] = np.nan
        q6_df[cat] = q6_df[cat].fillna(fill_means[cat])
    return q6_df[Q6_CATEGORIES].to_numpy(dtype=float)


def build_q6_features(df, fill_means, mu, sigma):
    raw = _raw_q6_matrix(df, fill_means)
    return add_bias((raw - mu) / sigma)


def fit_q6_stats(df, train_idx):
    train_df = df.iloc[train_idx]
    parsed = [_parse_q6_string(v) for v in train_df['Q6']]
    q6_train_df = pd.DataFrame(parsed, index=train_df.index)
    fill_means = {cat: (q6_train_df[cat].mean() if cat in q6_train_df.columns else 3.5) for cat in Q6_CATEGORIES}
    raw_train = _raw_q6_matrix(train_df, fill_means)
    mu = raw_train.mean(axis=0)
    sigma = raw_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    return fill_means, mu, sigma


# Q7: temperature guess -> cleaned, clipped, standardized scalar
def _clean_q7_value(value):
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def build_q7_features(df, median_fill, mu, sigma, lower=-30, upper=45):
    q7 = df['Q7'].apply(_clean_q7_value)
    q7 = q7.clip(lower, upper)
    q7 = q7.fillna(median_fill)
    x = q7.to_numpy(dtype=float).reshape(-1, 1)
    x_std = (x - mu) / sigma
    return add_bias(x_std)


def fit_q7_stats(df, train_idx, lower=-30, upper=45):
    q7 = df['Q7'].apply(_clean_q7_value).clip(lower, upper)
    train_vals = q7.iloc[train_idx]
    median_fill = train_vals.median()
    q7_filled = q7.fillna(median_fill)
    train_filled = q7_filled.iloc[train_idx]
    mu, sigma = train_filled.mean(), train_filled.std()
    return median_fill, mu, sigma


# Q10: free-text quote -> 39-dimension city-keyword indicator block (no bias;
# used by the decision tree, which handles its own thresholding)
def build_q10_features(df):
    q10 = df['Q10'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    feats = np.zeros((len(df), len(Q10_FEATURE_NAMES)), dtype=float)
    for j, kw in enumerate(Q10_FEATURE_NAMES):
        pattern = r'\b' + re.escape(kw) + r'\b'
        feats[:, j] = q10.str.contains(pattern, regex=True).to_numpy(dtype=float)
    return feats


# Softmax multiclass logistic regression
def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(X, y_one_hot, weights):
    probs = np.clip(softmax(X @ weights), 1e-15, 1 - 1e-15)
    return -np.sum(y_one_hot * np.log(probs)) / X.shape[0]


def to_one_hot(labels, city_order):
    mapping = {c: i for i, c in enumerate(city_order)}
    one_hot = np.zeros((len(labels), len(city_order)))
    for i, lab in enumerate(labels):
        one_hot[i, mapping[lab]] = 1.0
    return one_hot


def train_logreg(X_train, y_oh, X_val, y_val_oh, alpha=0.5, epochs=500, mode='batch', decay_rate=None, seed=None):
    rng = np.random.default_rng(seed)
    N, D = X_train.shape
    K = y_oh.shape[1]
    W = np.zeros((D, K))
    train_hist, val_hist = [], []
    lr = alpha
    for _ in range(epochs):
        if mode == 'batch':
            probs = softmax(X_train @ W)
            grad = X_train.T @ (probs - y_oh) / N
            W -= lr * grad
        else:  # sgd
            for idx in rng.permutation(N):
                xi, yi = X_train[idx:idx+1], y_oh[idx:idx+1]
                probs = softmax(xi @ W)
                grad = xi.T @ (probs - yi)
                W -= lr * grad
            if decay_rate:
                lr *= decay_rate
        train_hist.append(cross_entropy_loss(X_train, y_oh, W))
        val_hist.append(cross_entropy_loss(X_val, y_val_oh, W))
    return W, train_hist, val_hist


def logreg_proba(X, W):
    return softmax(X @ W)


def accuracy(pred_labels, true_labels):
    return float(np.mean(np.array(pred_labels) == np.array(true_labels)) * 100)


def tree_predict_proba(X, feature, threshold, children_left, children_right, value, classes):
    """value has shape (n_nodes, 1, n_classes) as stored by sklearn."""
    out = np.zeros((X.shape[0], len(classes)))
    for i, x in enumerate(X):
        node = 0
        while children_left[node] != children_right[node]:
            if x[feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        counts = value[node][0]
        out[i] = counts / counts.sum()
    return out
