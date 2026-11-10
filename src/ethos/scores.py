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
