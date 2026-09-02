"""
Combines five independently-trained sub-models (Q1-4, Q5, Q6, Q7, Q10),
each from a numpy softmax logistic regression (or, for Q10, a decision tree),
by averaging their class-probability outputs.

Q8 and Q9 are deliberately excluded: they were judged too subjective/noisy 
to contribute a reliable signal.

All learned parameters live in the small companion file `ensemble_weights.npz`
"""
import numpy as np
import pandas as pd

_this_file = __file__.replace('\\', '/')
_HERE = _this_file.rsplit('/', 1)[0] if '/' in _this_file else '.'
_WEIGHTS_PATH = _HERE + '/ensemble_weights.npz'

# Fixed feature orderings (must match what the weights were trained with)
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


def _add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _build_q1_4(df, q1f, q2f, q3f, q4f, mu, sigma):
    q1 = df['Q1'].fillna(q1f).to_numpy(dtype=float) if 'Q1' in df else np.full(len(df), q1f)
    q2 = df['Q2'].fillna(q2f).to_numpy(dtype=float) if 'Q2' in df else np.full(len(df), q2f)
    q3 = df['Q3'].fillna(q3f).to_numpy(dtype=float) if 'Q3' in df else np.full(len(df), q3f)
    q4 = df['Q4'].fillna(q4f).to_numpy(dtype=float) if 'Q4' in df else np.full(len(df), q4f)
    poly = np.column_stack([
        q1, q2, q3, q4,
        q1**2, q2**2, q3**2, q4**2,
        q1*q2, q1*q3, q1*q4,
        q2*q3, q2*q4, q3*q4,
    ])
    return _add_bias((poly - mu) / sigma)


def _build_q5(df):
    q5 = df['Q5'].fillna('Unknown').astype(str) if 'Q5' in df else pd.Series(['Unknown'] * len(df))
    q5_clean = q5.str.replace(r'\s*,\s*', ',', regex=True)
    dummies = q5_clean.str.get_dummies(sep=',')
    for cat in Q5_CATEGORIES:
        if cat not in dummies.columns:
            dummies[cat] = 0
    dummies = dummies[Q5_CATEGORIES]
    return _add_bias(dummies.to_numpy(dtype=float))


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


def _build_q6(df, fill_arr, mu, sigma):
    fill_means = dict(zip(Q6_CATEGORIES, fill_arr))
    col = df['Q6'] if 'Q6' in df else pd.Series([None] * len(df))
    parsed = [_parse_q6_string(v) for v in col]
    q6_df = pd.DataFrame(parsed, index=df.index)
    for cat in Q6_CATEGORIES:
        if cat not in q6_df.columns:
            q6_df[cat] = np.nan
        q6_df[cat] = q6_df[cat].fillna(fill_means[cat])
    raw = q6_df[Q6_CATEGORIES].to_numpy(dtype=float)
    return _add_bias((raw - mu) / sigma)


def _clean_q7_value(value):
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _build_q7(df, median_fill, mu, sigma, lower=-30, upper=45):
    col = df['Q7'] if 'Q7' in df else pd.Series([None] * len(df))
    q7 = col.apply(_clean_q7_value).clip(lower, upper).fillna(median_fill)
    x = q7.to_numpy(dtype=float).reshape(-1, 1)
    return _add_bias((x - mu) / sigma)


def _build_q10(df):
    col = df['Q10'] if 'Q10' in df else pd.Series([''] * len(df))
    q10 = col.fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    feats = np.zeros((len(df), len(Q10_FEATURE_NAMES)), dtype=float)
    for j, kw in enumerate(Q10_FEATURE_NAMES):
        pattern = r'\b' + kw + r'\b'
        feats[:, j] = q10.str.contains(pattern, regex=True).to_numpy(dtype=float)
    return feats


def _tree_predict_proba(X, feature, threshold, children_left, children_right, value, n_classes):
    out = np.zeros((X.shape[0], n_classes))
    for i in range(X.shape[0]):
        node = 0
        x = X[i]
        while children_left[node] != children_right[node]:
            if x[feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        counts = value[node][0]
        out[i] = counts / counts.sum()
    return out

# Load trained parameters once at import time
_W = np.load(_WEIGHTS_PATH, allow_pickle=True)
_CITY_ORDER = [str(c) for c in _W['city_order']]
_TREE_CLASSES = [str(c) for c in _W['tree_classes']]
_TREE_REORDER = [_TREE_CLASSES.index(c) for c in _CITY_ORDER]


def predict(row_df):
    """row_df: a pandas DataFrame (any number of rows) with the raw survey columns."""
    X_q1_4 = _build_q1_4(
        row_df, float(_W['q1_fill']), float(_W['q2_fill']), float(_W['q3_fill']), float(_W['q4_fill']),
        _W['q1_4_mu'], _W['q1_4_sigma'],
    )
    X_q5 = _build_q5(row_df)
    X_q6 = _build_q6(row_df, _W['q6_fill'], _W['q6_mu'], _W['q6_sigma'])
    X_q7 = _build_q7(row_df, float(_W['q7_median']), float(_W['q7_mu']), float(_W['q7_sigma']))
    X_q10 = _build_q10(row_df)

    p_q1_4 = _softmax(X_q1_4 @ _W['W_q1_4'])
    p_q5 = _softmax(X_q5 @ _W['W_q5'])
    p_q6 = _softmax(X_q6 @ _W['W_q6'])
    p_q7 = _softmax(X_q7 @ _W['W_q7'])
    p_q10_raw = _tree_predict_proba(
        X_q10,
        feature=_W['tree_feature'], threshold=_W['tree_threshold'],
        children_left=_W['tree_children_left'], children_right=_W['tree_children_right'],
        value=_W['tree_value'], n_classes=len(_TREE_CLASSES),
    )
    p_q10 = p_q10_raw[:, _TREE_REORDER]

    use_learned = bool(_W['use_learned_meta'])
    if use_learned:
        stacked = np.hstack([p_q1_4, p_q5, p_q6, p_q7, p_q10])
        stacked_b = _add_bias(stacked)
        final_probs = _softmax(stacked_b @ _W['meta_W'])
    else:
        final_probs = np.mean([p_q1_4, p_q5, p_q6, p_q7, p_q10], axis=0)

    pred_idx = final_probs.argmax(axis=1)
    return [_CITY_ORDER[i] for i in pred_idx]


def predict_all(filename):
    """
    Make predictions for the data in filename.
    Returns a list of city-name predictions, one per row, in file order.
    """
    df = pd.read_csv(filename)
    return predict(df)
