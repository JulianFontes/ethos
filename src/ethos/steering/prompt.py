"""Prompt steering: a value or moral profile in the judge's system prompt.

prompt_conditions  {condition name: profile text}; "none" (empty text) = no profile
system_prompt      the system prompt for a profile text (the minimal judge prompt when there is no profile)
condition_name     a condition's name, tagged when the answers also come from an activation-steered model
"""
from functools import cache
from pathlib import Path

import yaml

from ethos.judges.llm_judge import load_profile
from ethos.llm.dmp import format_profile
from ethos.paths import ROOT

BASE_PROMPT = ROOT / "prompts/judge_minimal.txt"                # no profile
DMP_PROMPT = ROOT / "prompts/judge_minimal_pluralistic.txt"     # "... with the following moral profile: {profile}"
WEIGHTINGS = ["as-is", "equal", "reversed"]


def prompt_conditions(taxonomy, weightings=WEIGHTINGS, single_values=False, profiles=Path("prompts/profiles"),
                      any_values=False):
    """{condition name: profile text}; "none" (empty text) = no profile.
    single_values: each taxonomy value alone at weight 1 ("<value>|as-is").
    Otherwise every profile YAML file in `profiles` ("<file stem>|<weighting>"), under each of `weightings`:
      as-is     the file's weights
      equal     every value the same weight
      reversed  the file's weights in reverse order
    Profile values must be in `taxonomy` unless `any_values` (e.g. for profiles from another framework)."""
    out = {"none": ""}
    if single_values:
        return out | {f"{v}|as-is": format_profile([(v, 1.0)]) for v in taxonomy}
    paths = sorted((ROOT / profiles).glob("*.yaml"))
    assert paths, f"no profiles (*.yaml) in {profiles}"
    for path in paths:
        names = list(yaml.safe_load(path.read_text())["values"]) if any_values else taxonomy
        profile = load_profile(path.relative_to(ROOT), names)
        values, weights = [v for v, _ in profile], [w for _, w in profile]
        variants = {"as-is": profile, "equal": [(v, 1 / len(values)) for v in values],
                    "reversed": list(zip(values, weights[::-1]))}
        for w in weightings:
            out[f"{path.stem}|{w}"] = format_profile(variants[w])
    return out


@cache
def _template(path):
    return path.read_text().strip()

def system_prompt(profile_text):
    """The judge's system prompt: the pluralistic prompt with the profile filled in, or the minimal one without."""
    if not profile_text:
        return _template(BASE_PROMPT)
    return _template(DMP_PROMPT).replace("{profile}", profile_text)

def condition_name(prompt_condition, steer_tag=None):
    """"Justice|as-is" -> "Justice|as-is|steer=justice@+4" when the answering model is activation-steered."""
    return f"{prompt_condition}|steer={steer_tag}" if steer_tag else prompt_condition
