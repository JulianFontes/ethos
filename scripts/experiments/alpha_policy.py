"""Train and evaluate the steering-strength policy (ethos.alpha_policy) with k-fold cross-validation, one policy per
input set. Every setting is in the config: run, out, model (bell / softmax / multibell), features, folds, epochs, seed,
and optionally baseline: a value (e.g. toxicity) whose options alone also train a bell policy per features entry, on
the same folds, drawn next to the main policies and compared with them comment by comment.

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
import torch
import yaml

from ethos.alpha_policy import (cross_validate, load, option_value, paired_difference, prior_curve,
                                rationale_embeddings)
from ethos.data import load_steering_scores
from ethos.paths import ROOT
from ethos.plotting.alpha_policy import (METHODS, plot_error_by_band, plot_loss, plot_policy_curve, plot_toxic_share,
                                         plot_toxic_share_by_band, style)


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
    baseline = cfg.get("baseline")
    if baseline:                       # the single-value bell policy on the same comments, folds and features
        cols = [i for i, o in enumerate(options) if option_value(o) in (None, baseline)]
        base, _ = cross_validate(X, D[:, cols], S[:, cols], [options[i] for i in cols], methods, R, model="bell",
                                 folds=folds, epochs=epochs, seed=seed)
        cols = torch.tensor(cols)      # its picks index its own options: map them to the run's
        results |= {f"{baseline} only: {m}": (h, cols[oof]) for m, (h, oof) in base.items()}
    pd.concat([h.assign(method=m) for m, (h, _) in results.items()], ignore_index=True).to_csv(out / "history.csv",
                                                                                                index=False)

    chosen = pd.Series(best["choice"]).value_counts()
    several = len({option_value(o) for o in options} - {None}) > 1     # then name the value too
    label = lambda o: ("0 (no steering)" if o == "none" else
                       f"{option_value(o)} {o.rsplit('@', 1)[1]}" if several else o.rsplit("@", 1)[1])
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
        print(f"  {style(method)[0]:62s} final {v.iloc[-1]:.4f}   lowest {v.min():.4f} at epoch {v.idxmin()}")
    if baseline:
        err = lambda oof: D.gather(1, oof[-1][:, None]).squeeze(1).numpy()
        print(f"{model} minus {baseline} only, per comment (95% bootstrap interval; < 0: {model} closer to humans):")
        for m in methods:
            d, lo, hi = paired_difference(err(results[m][1]), err(results[f"{baseline} only: {m}"][1]))
            picks = pd.Series([option_value(options[i]) or "none" for i in results[m][1][-1]]).value_counts(normalize=True)
            print(f"  {METHODS[m][0]:46s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   picks: "
                  + ", ".join(f"{v} {p:.0%}" for v, p in picks.items()))
    print(f"wrote {cfg['out']}/: history.csv, loss.png, toxic_share.png, toxic_share_by_band.png, error_by_band.png"
          + (", policy_curve.png" if curve else ""))


if __name__ == "__main__":
    main()
