"""The LLM toxicity score: the share of an LLM's answers that are TOXIC.

Defined like the human score (the share of annotators who rated a comment toxic): the LLM's repeated answers to one
comment act as its panel of annotators. Every judge, experiment and plot computes the score through this function.
"""
import pandas as pd


def _is_toxic(values):
    """Series of answers -> bool Series. Accepts True/False or "TOXIC"/"NOT TOXIC" (any case)."""
    s = pd.Series(values)
    if s.dtype == bool:
        return s
    return s.map(lambda v: str(v).strip().upper() in ("TOXIC", "TRUE"))


def toxicity_score(answers, by=None):
    """Share of answers that are TOXIC.

    answers: a list of answers (True/False or "TOXIC"/"NOT TOXIC"), or a DataFrame with one row per answer and a
    `toxic` (True/False) or `evaluation` ("TOXIC"/"NOT TOXIC") column.
    by: with a DataFrame, the column(s) to score separately, e.g. "id" (one score per comment) or ["id", "condition"].

    Returns a float, or with `by` a Series indexed by those columns.
    """
    if not isinstance(answers, pd.DataFrame):
        return float(_is_toxic(list(answers)).mean())
    toxic = _is_toxic(answers["toxic" if "toxic" in answers else "evaluation"]).astype(float)
    if by is None:
        return float(toxic.mean())
    keys = [by] if isinstance(by, str) else list(by)
    return toxic.groupby([answers[k] for k in keys]).mean()


def score_table(ratings, sample):
    """One row per comment and condition: the LLM toxicity score next to the human score.

    ratings: answers from scripts/experiments/profile_signal.py (id, condition, toxic); sample: its sample.csv.
    Columns: id, comment, moral_value ("none" = no value in the prompt; profiles keep their file name), weighting,
    llm_toxicity, human_toxicity, n_answers, consensus, consensus_bin.
    Activation steering runs (no-value prompt, conditions "none|steer=<pole>@<alpha>"): moral_value is the pole the
    vector pushes toward (e.g. "toxicity", "not_toxicity"; "none" unsteered) and weighting the alpha (0 unsteered).
    """
    scores = toxicity_score(ratings, by=["id", "condition"]).rename("llm_toxicity")
    n = ratings.groupby(["id", "condition"]).size().rename("n_answers")
    t = pd.concat([scores, n], axis=1).reset_index()
    split = t.condition.str.split("|", n=1, expand=True)
    t["moral_value"], t["weighting"] = split[0], split[1] if 1 in split else None
    t = steering_columns(t)
    s = sample.set_index("id")
    t = t.join(s[["comment_text", "toxicity", "consensus", "consensus_bin"]], on="id")
    t = t.rename(columns={"comment_text": "comment", "toxicity": "human_toxicity"})
    cols = ["id", "comment", "moral_value", "weighting", "llm_toxicity", "human_toxicity", "n_answers", "consensus",
            "consensus_bin"]
    return t[cols].sort_values(["id", "moral_value", "weighting"], na_position="first", ignore_index=True)


def steering_columns(t):
    """For activation steering rows (moral_value "none", weighting "steer=<pole>@<alpha>"): moral_value = the pole,
    weighting = alpha (a number); the unsteered rows of a steering table get weighting 0. Other tables are returned
    unchanged. Also converts scores.csv files written before this format."""
    w = t.weighting.astype("string")
    steered = w.str.startswith("steer=", na=False)
    if not steered.any():
        return t
    assert (t.moral_value[steered] == "none").all(), "steering with a value in the prompt: keep the old columns"
    pole_alpha = w[steered].str.removeprefix("steer=").str.rsplit("@", n=1, expand=True)
    t = t.copy()
    t["weighting"] = pd.Series(0.0, index=t.index).where(~steered, pd.to_numeric(pole_alpha[1]))
    t.loc[steered, "moral_value"] = pole_alpha[0]
    assert t.weighting.notna().all() and (t.moral_value[~steered] == "none").all()
    return t
