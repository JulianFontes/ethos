"""Train and evaluate the steering-strength policy (ethos.alpha_policy) with k-fold cross-validation, one policy per
input set. Every setting is in the config: run, out, model (bell / softmax), features, folds, epochs, seed.

Writes to `out`:
  history.csv               train / validation error per policy, fold and epoch
  loss.png                  validation error per epoch (mean +- sd over folds), and the CDF of the policies' scores
  toxic_share.png           toxic vs not toxic: humans, the LLM unsteered, each policy
  toxic_share_by_band.png   the same per consensus band
  error_by_band.png         average |LLM - human| per consensus band (95% bootstrap intervals): unsteered, each policy
  policy_curve.png          (bell, with the prior-score policy) the alpha it aims for at each prior score

Usage: python scripts/experiments/alpha_policy.py --config configs/llama3_8b/alpha_policy.yaml
       (scripts/shell/run_alpha_policy.sh)
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from ethos.alpha_policy import cross_validate, load, prior_curve, rationale_embeddings
from ethos.data import load_steering_scores
from ethos.paths import ROOT
from ethos.plotting.alpha_policy import (METHODS, plot_error_by_band, plot_loss, plot_policy_curve, plot_toxic_share,
                                         plot_toxic_share_by_band)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/llama3_8b/alpha_policy.yaml"))
    cfg = yaml.safe_load((ROOT / p.parse_args().config).read_text())
    scores = Path(cfg["run"]) / "scores.csv"
    model, folds, epochs, seed = cfg.get("model", "bell"), cfg.get("folds", 5), cfg.get("epochs", 300), cfg.get("seed", 0)
    methods = cfg.get("features", list(METHODS))
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)

    X, D, S, H, options = load(scores, complete_only=True)
    rows, _ = load_steering_scores(scores, complete_only=True)
    ids = sorted(rows.id.unique())                                        # the order of load's pivots
    R = rationale_embeddings(scores, ids)
    if R is None and any("rationale" in m for m in methods):
        methods = [m for m in methods if "rationale" not in m]
        print("no rationale_embeddings.parquet in the run: skipping the rationale policies")
    none = options.index("none")

    results, best = cross_validate(X, D, S, options, methods, R, model=model, folds=folds, epochs=epochs, seed=seed)
    pd.concat([h.assign(method=m) for m, (h, _) in results.items()], ignore_index=True).to_csv(out / "history.csv",
                                                                                                index=False)

    chosen = pd.Series(best["choice"]).value_counts()
    label = lambda o: "0 (no steering)" if o == "none" else o.rsplit("@", 1)[1]
    best_name = "best constant α " + (label(chosen.index[0]) if len(chosen) == 1 else
                                       " / ".join(f"{label(o)} ({n} folds)" for o, n in chosen.items()))
    constants = {"none": (D[:, none].mean().item(), S[:, none], "α 0 (no steering)", "#555", "-"),
                 "best": (best["error"], best["scores"], best_name, "#e08a2e", "--")}
    bands = rows.groupby("id").consensus_bin.first().astype(str).tolist()      # sorted by id, like the pivots in load
    n = len(X)
    plot_loss(results, S, H, constants, out / "loss.png", f"Steering Strength Policy: {n} comments, {folds}-fold "
              "cross-validation")
    plot_toxic_share(results, S, H, none, out / "toxic_share.png", f"Toxic vs Not Toxic: {n} comments")
    plot_toxic_share_by_band(results, S, H, none, bands, out / "toxic_share_by_band.png",
                             f"Toxic vs Not Toxic by Consensus Band: {n} comments")
    plot_error_by_band(results, D, none, bands, out / "error_by_band.png",
                       f"Model-Human Alignment Across Consensus Bands: {n} comments")
    curve = model == "bell" and "prior" in methods
    if curve:
        plot_policy_curve(prior_curve(X, D, S, options, epochs=epochs, seed=seed), D, S, options,
                          out / "policy_curve.png", "Steering Strength Policy: What It Picks by Prior Score", seed=seed)

    print(f"validation error using the top alpha (mean over folds); {best_name}: {best['error']:.4f}, "
          f"alpha 0: {D[:, none].mean().item():.4f}")
    for method, (history, _) in results.items():
        v = history.groupby("epoch")["validation, top alpha"].mean()
        print(f"  {METHODS[method][0]:46s} final {v.iloc[-1]:.4f}   lowest {v.min():.4f} at epoch {v.idxmin()}")
    print(f"wrote {cfg['out']}/: history.csv, loss.png, toxic_share.png, toxic_share_by_band.png, error_by_band.png"
          + (", policy_curve.png" if curve else ""))


if __name__ == "__main__":
    main()
