"""Value policy: for each comment, choose the one moral value whose value-conditioned LLM lands closest to the human
toxicity score. Training data comes from ethos.data.load_scores / feature_matrix.

  cv_splits(rows)                   cross-validation rounds that split by comment, balanced by consensus band
  best_fixed_value(rows, idx)       the single value that is best on average on some comments
  baselines(rows, base, idx, v)     mean distance of the no-value, random, fixed-value and oracle policies
  run_tree_cv(rows, X, depths, ...) decision tree policy: fit per round and depth, pick the lowest predicted value
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS, N_REPEATS = 5, 5

def cv_splits(rows, n_folds=N_FOLDS, n_repeats=N_REPEATS):
    """Cross-validation rounds over `rows` (one row per comment and value, from load_scores).

    Each comment's rows stay together (all in training or all in validation), and each fold gets the same mix of
    consensus bands. Repeated with `n_repeats` different shuffles. Yields (repeat, fold, train_idx, val_idx), where the
    indices are row positions in `rows`.
    """
    groups, bands = rows.id.to_numpy(), rows.consensus_bin.to_numpy()
    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=repeat)
        for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(rows)), bands, groups)):
            yield repeat, fold, train_idx, val_idx


def baselines(rows, base, idx, fixed_value):
    """Mean distance from the human score on the comments of rows.iloc[idx], for each baseline policy:

    no_value     the LLM with no value in the prompt (base: the no-value rows from load_scores)
    random       a random value per comment (in expectation: the mean over all 60 values)
    fixed_value  the same value for every comment (`fixed_value`, chosen on the training comments)
    oracle       each comment's best value, chosen by looking at its distances (a ceiling, not a policy)
    """
    r = rows.iloc[idx]
    ids = r.id.unique()
    per_comment = r.groupby("id").distance
    return {"no_value": base.loc[ids, "distance"].mean(),
            "random": per_comment.mean().mean(),
            "fixed_value": r[r.moral_value == fixed_value].distance.mean(),
            "oracle": per_comment.min().mean()}


def best_fixed_value(rows, idx):
    """The value with the lowest mean distance on the comments of rows.iloc[idx]."""
    return rows.iloc[idx].groupby("moral_value").distance.mean().idxmin()


# --- decision tree policy --------------------------------------------------------------------------------------

def choose(rows, predicted, tiebreak):
    """For each comment in `rows`, the value with the lowest predicted distance. Ties (several values with the same
    prediction, common in shallow trees) go to the value with the lower `tiebreak` (its mean distance on the training
    comments). Returns (picks: Series id -> value, tied: Series id -> True if the lowest prediction was shared)."""
    df = rows[["id", "moral_value"]].assign(pred=predicted, tb=rows.moral_value.map(tiebreak).to_numpy())
    lowest = df.groupby("id").pred.transform("min")
    tied = (df.pred == lowest).groupby(df.id).sum() > 1
    picks = df.sort_values(["id", "pred", "tb"]).groupby("id").head(1).set_index("id").moral_value
    return picks, tied


def picked_distance(rows, picks):
    """Mean real distance from the human score of the picked values (one per comment)."""
    keyed = rows.set_index(["id", "moral_value"]).distance
    return keyed.loc[list(picks.items())].mean()


def run_tree_cv(rows, X, depths, splits, progress=None, on_round=None):
    """Fit a DecisionTreeRegressor (predicting distance) for every round in `splits` and every depth.

    Returns one record per round and depth: train/val MSE of the predicted distance, the policy's real distance on
    train and val comments, the share of val comments with tied predictions, and the tree's leaf count.
    `progress` (a tqdm bar) is advanced once per round and shows the running mean validation distance per depth.
    `on_round(results so far)` is called after each round, e.g. to save partial results while the run goes on.
    """
    from sklearn.tree import DecisionTreeRegressor

    records, y = [], rows.distance.to_numpy()
    for repeat, fold, tr, va in splits:
        tiebreak = rows.iloc[tr].groupby("moral_value").distance.mean()
        for depth in depths:
            tree = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X.iloc[tr], y[tr])
            rec = {"repeat": repeat, "fold": fold, "depth": depth, "leaves": tree.get_n_leaves()}
            for split, idx in (("train", tr), ("val", va)):
                pred = tree.predict(X.iloc[idx])
                picks, tied = choose(rows.iloc[idx], pred, tiebreak)
                rec[f"{split}_mse"] = float(((pred - y[idx]) ** 2).mean())
                rec[f"{split}_policy"] = picked_distance(rows.iloc[idx], picks)
                rec[f"{split}_tied"] = float(tied.mean())
            records.append(rec)
        if progress is not None:
            done = pd.DataFrame(records).groupby("depth").val_policy.mean()
            progress.set_postfix(best=f"depth {done.idxmin()}: {done.min():.3f}")
            progress.update(1)
        if on_round is not None:
            on_round(pd.DataFrame(records))
    return pd.DataFrame(records)
