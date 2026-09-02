import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from feature_lib import (
    build_q1_4_features, fit_q1_4_stats,
    build_q5_features,
    build_q6_features, fit_q6_stats,
    build_q7_features, fit_q7_stats,
    build_q10_features,
    to_one_hot, train_logreg, logreg_proba, accuracy,
    tree_predict_proba, add_bias,
)

RNG_SEED = 1024

# ------------------------------------------------------------------
# 1. Reproduce the EXACT split every sub-model notebook used
#    global_train_idx (70%) -> train_idx/val_idx (80/20 of that) used
#    for sub-model hyperparameter tuning; global_val_idx (10%) reserved
#    for the late-fusion meta-model; global_test_idx (20%) reserved for
#    a single final unbiased report.
# ------------------------------------------------------------------
df = pd.read_csv('cleaned_dataset.csv')
df = df.dropna(subset=['Label']).reset_index(drop=True)
df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)

master_idx = np.arange(len(df))
labels = df['Label']

train_meta_idx, global_test_idx, y_train_meta, y_test = train_test_split(
    master_idx, labels, test_size=0.20, stratify=labels, random_state=RNG_SEED
)
global_train_idx, global_val_idx, y_train, y_val = train_test_split(
    train_meta_idx, y_train_meta, test_size=0.125, stratify=y_train_meta, random_state=RNG_SEED
)
labels_train_pool = labels[global_train_idx]
train_idx, val_idx, y_sub_train, y_sub_val = train_test_split(
    global_train_idx, labels_train_pool, test_size=0.20, stratify=labels_train_pool, random_state=RNG_SEED
)

print(f"train_idx: {len(train_idx)}  val_idx: {len(val_idx)}  "
      f"global_train_idx: {len(global_train_idx)}  global_val_idx: {len(global_val_idx)}  "
      f"global_test_idx: {len(global_test_idx)}")

city_order = sorted(df['Label'].unique().tolist())
print("City order:", city_order)
y_full = df['Label'].to_numpy()

# ------------------------------------------------------------------
# 2a. Fit preprocessing stats on train_idx ONLY (for hyperparam search)
# ------------------------------------------------------------------
q1_4_fill_s, q1_4_mu_s, q1_4_sigma_s = fit_q1_4_stats(df, train_idx)
q6_fill_s, q6_mu_s, q6_sigma_s = fit_q6_stats(df, train_idx)
q7_median_s, q7_mu_s, q7_sigma_s = fit_q7_stats(df, train_idx)

X_q1_4_s = build_q1_4_features(df, q1_4_fill_s, q1_4_mu_s, q1_4_sigma_s)
X_q5_s = build_q5_features(df)
X_q6_s = build_q6_features(df, q6_fill_s, q6_mu_s, q6_sigma_s)
X_q7_s = build_q7_features(df, q7_median_s, q7_mu_s, q7_sigma_s)
X_q10_s = build_q10_features(df)

y_oh_train_s = to_one_hot(y_full[train_idx], city_order)
y_oh_val_s = to_one_hot(y_full[val_idx], city_order)

# ------------------------------------------------------------------
# 2b. Hyperparameter search per sub-model (alpha, batch vs sgd) using
#     train_idx -> val_idx only.
# ------------------------------------------------------------------
search_space = {
    'Q1_4': X_q1_4_s,
    'Q5': X_q5_s,
    'Q6': X_q6_s,
    'Q7': X_q7_s,
}
alphas = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
chosen_hparams = {}

print("\n--- Sub-model hyperparameter search (train_idx -> val_idx) ---")
for name, X in search_space.items():
    best = None
    for mode in ['batch', 'sgd']:
        for alpha in alphas:
            W, _, _ = train_logreg(
                X[train_idx], y_oh_train_s, X[val_idx], y_oh_val_s,
                alpha=alpha, epochs=500, mode=mode,
                decay_rate=0.98 if mode == 'sgd' else None, seed=RNG_SEED,
            )
            val_probs = logreg_proba(X[val_idx], W)
            val_acc = accuracy([city_order[i] for i in val_probs.argmax(1)], y_full[val_idx])
            if best is None or val_acc > best[2]:
                best = (mode, alpha, val_acc)
    print(f"  [{name}] best: mode={best[0]} alpha={best[1]} val_acc={best[2]:.2f}%")
    chosen_hparams[name] = dict(mode=best[0], alpha=best[1], epochs=500)

# Q10 tree depth search (train_idx -> val_idx)
X_q10_train, X_q10_val = X_q10_s[train_idx], X_q10_s[val_idx]
y_q10_train, y_q10_val = y_full[train_idx], y_full[val_idx]
best_depth, best_depth_acc = None, -1
for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, None]:
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, min_samples_split=2, random_state=RNG_SEED)
    tree.fit(X_q10_train, y_q10_train)
    val_acc = tree.score(X_q10_val, y_q10_val) * 100
    if val_acc > best_depth_acc:
        best_depth, best_depth_acc = depth, val_acc
print(f"  [Q10] best: max_depth={best_depth} val_acc={best_depth_acc:.2f}%")

# ------------------------------------------------------------------
# 3. Refit final sub-models on the FULL 70% pool (global_train_idx),
#    using the hyperparameters chosen above and stats fit on that pool.
# ------------------------------------------------------------------
q1_4_fill, q1_4_mu, q1_4_sigma = fit_q1_4_stats(df, global_train_idx)
q6_fill, q6_mu, q6_sigma = fit_q6_stats(df, global_train_idx)
q7_median, q7_mu, q7_sigma = fit_q7_stats(df, global_train_idx)

X_q1_4 = build_q1_4_features(df, q1_4_fill, q1_4_mu, q1_4_sigma)
X_q5 = build_q5_features(df)
X_q6 = build_q6_features(df, q6_fill, q6_mu, q6_sigma)
X_q7 = build_q7_features(df, q7_median, q7_mu, q7_sigma)
X_q10 = build_q10_features(df)

FEATURES = {'Q1_4': X_q1_4, 'Q5': X_q5, 'Q6': X_q6, 'Q7': X_q7}

y_oh_gtrain = to_one_hot(y_full[global_train_idx], city_order)
y_oh_gval = to_one_hot(y_full[global_val_idx], city_order)
y_oh_gtest = to_one_hot(y_full[global_test_idx], city_order)

print("\n--- Final refit on global_train_idx (70%), evaluated on global_val_idx (10%, held out) ---")
logreg_weights = {}
for name, X in FEATURES.items():
    hp = chosen_hparams[name]
    W, _, _ = train_logreg(
        X[global_train_idx], y_oh_gtrain, X[global_val_idx], y_oh_gval,
        alpha=hp['alpha'], epochs=hp['epochs'], mode=hp['mode'],
        decay_rate=0.98 if hp['mode'] == 'sgd' else None, seed=RNG_SEED,
    )
    logreg_weights[name] = W
    val_probs = logreg_proba(X[global_val_idx], W)
    val_acc = accuracy([city_order[i] for i in val_probs.argmax(1)], y_full[global_val_idx])
    print(f"  [{name}] {hp} -> global_val acc={val_acc:.2f}%")

final_tree = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, min_samples_split=2, random_state=RNG_SEED)
final_tree.fit(X_q10[global_train_idx], y_full[global_train_idx])
tree_val_acc = final_tree.score(X_q10[global_val_idx], y_full[global_val_idx]) * 100
print(f"  [Q10] max_depth={best_depth} -> global_val acc={tree_val_acc:.2f}%")

t_ = final_tree.tree_
q10_classes = final_tree.classes_.tolist()
q10_tree_arrays = dict(
    feature=t_.feature.copy(), threshold=t_.threshold.copy(),
    children_left=t_.children_left.copy(), children_right=t_.children_right.copy(),
    value=t_.value.copy(),
)

# ------------------------------------------------------------------
# 4. Build meta-training features: each sub-model's class-probability
#    vector on global_val_idx (147 rows, never used for training OR
#    hyperparameter tuning by any sub-model -- genuine out-of-fold data)
# ------------------------------------------------------------------
MODEL_ORDER = ['Q1_4', 'Q5', 'Q6', 'Q7', 'Q10']


def submodel_probs(idx):
    out = {}
    for name, W in logreg_weights.items():
        out[name] = logreg_proba(FEATURES[name][idx], W)
    raw = tree_predict_proba(X_q10[idx], classes=q10_classes, **q10_tree_arrays)
    reorder = [q10_classes.index(c) for c in city_order]
    out['Q10'] = raw[:, reorder]
    return out


def stack(prob_dict):
    return np.hstack([prob_dict[m] for m in MODEL_ORDER])


probs_gval = submodel_probs(global_val_idx)
probs_gtest = submodel_probs(global_test_idx)

X_meta_val_b = add_bias(stack(probs_gval))
X_meta_test_b = add_bias(stack(probs_gtest))

# small internal split of global_val_idx to pick the meta-model's alpha
meta_tr, meta_tune = train_test_split(
    np.arange(len(global_val_idx)), test_size=0.3, stratify=y_full[global_val_idx], random_state=RNG_SEED
)
y_meta_oh_full = to_one_hot(y_full[global_val_idx], city_order)

print("\n--- Meta-fusion alpha search (global_val split internally) ---")
best_meta_alpha, best_meta_acc = None, -1
for alpha in [0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
    W, _, _ = train_logreg(
        X_meta_val_b[meta_tr], y_meta_oh_full[meta_tr],
        X_meta_val_b[meta_tune], y_meta_oh_full[meta_tune],
        alpha=alpha, epochs=800, mode='batch', seed=RNG_SEED,
    )
    tune_probs = logreg_proba(X_meta_val_b[meta_tune], W)
    tune_acc = accuracy([city_order[i] for i in tune_probs.argmax(1)], y_full[global_val_idx][meta_tune])
    print(f"  alpha={alpha}  meta-tune acc={tune_acc:.2f}%")
    if tune_acc > best_meta_acc:
        best_meta_alpha, best_meta_acc = alpha, tune_acc

print(f"Chosen meta alpha = {best_meta_alpha}")

# Fair comparison, WITHOUT looking at global_test_idx: does the learned
# meta-model actually beat plain uniform averaging on the tune split?
tune_stack = stack({k: v[meta_tune] for k, v in probs_gval.items()})
uniform_tune_preds = [city_order[i] for i in tune_stack.reshape(len(meta_tune), 5, 4).mean(1).argmax(1)]
uniform_tune_acc = accuracy(uniform_tune_preds, y_full[global_val_idx][meta_tune])
print(f"  Uniform-average on same tune split: {uniform_tune_acc:.2f}%  (learned best: {best_meta_acc:.2f}%)")
USE_LEARNED_META = best_meta_acc > uniform_tune_acc
print(f"  -> Using {'LEARNED meta-fusion' if USE_LEARNED_META else 'UNIFORM averaging'} as the final combiner "
      f"(decided on internal tune split, never on global_test_idx)")

meta_W, _, _ = train_logreg(
    X_meta_val_b, y_meta_oh_full, X_meta_test_b, to_one_hot(y_full[global_test_idx], city_order),
    alpha=best_meta_alpha, epochs=1500, mode='batch', seed=RNG_SEED,
)

# ------------------------------------------------------------------
# 5. FINAL, single, unbiased report on global_test_idx (20%, touched
#    by nothing above)
# ------------------------------------------------------------------
print("\n================ FINAL HOLD-OUT (20%, global_test_idx) ================")
y_test_true = y_full[global_test_idx]
for name in MODEL_ORDER:
    preds = [city_order[i] for i in probs_gtest[name].argmax(1)]
    print(f"  {name:6s} solo test accuracy: {accuracy(preds, y_test_true):.2f}%")

uniform_avg = np.mean([probs_gtest[m] for m in MODEL_ORDER], axis=0)
uniform_preds = [city_order[i] for i in uniform_avg.argmax(1)]
print(f"  Uniform-average ensemble test accuracy: {accuracy(uniform_preds, y_test_true):.2f}%")

meta_probs_test = logreg_proba(X_meta_test_b, meta_W)
meta_preds = [city_order[i] for i in meta_probs_test.argmax(1)]
print(f"  LEARNED late-fusion test accuracy: {accuracy(meta_preds, y_test_true):.2f}%")
final_choice = 'LEARNED meta-fusion' if USE_LEARNED_META else 'UNIFORM averaging'
final_acc = accuracy(meta_preds, y_test_true) if USE_LEARNED_META else accuracy(uniform_preds, y_test_true)
print(f"  >>> FINAL CHOSEN COMBINER: {final_choice}  ->  test accuracy = {final_acc:.2f}% <<<")
print("=========================================================================")

# ------------------------------------------------------------------
# 6. Persist everything pred.py needs
# ------------------------------------------------------------------
np.savez(
    'ensemble_weights.npz',
    city_order=np.array(city_order),
    model_order=np.array(MODEL_ORDER),
    W_q1_4=logreg_weights['Q1_4'], W_q5=logreg_weights['Q5'],
    W_q6=logreg_weights['Q6'], W_q7=logreg_weights['Q7'],
    q1_fill=q1_4_fill['Q1'], q2_fill=q1_4_fill['Q2'], q3_fill=q1_4_fill['Q3'], q4_fill=q1_4_fill['Q4'],
    q1_4_mu=q1_4_mu, q1_4_sigma=q1_4_sigma,
    q6_fill=np.array([q6_fill[c] for c in ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']]),
    q6_mu=q6_mu, q6_sigma=q6_sigma,
    q7_median=q7_median, q7_mu=q7_mu, q7_sigma=q7_sigma,
    tree_feature=q10_tree_arrays['feature'], tree_threshold=q10_tree_arrays['threshold'],
    tree_children_left=q10_tree_arrays['children_left'], tree_children_right=q10_tree_arrays['children_right'],
    tree_value=q10_tree_arrays['value'], tree_classes=np.array(q10_classes),
    meta_W=meta_W,
    use_learned_meta=np.array(USE_LEARNED_META),
)
import os
print("\nSaved ensemble_weights.npz — size (KB):", os.path.getsize('ensemble_weights.npz') / 1024)
