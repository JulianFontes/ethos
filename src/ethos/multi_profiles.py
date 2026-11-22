"""Multi-value profiles built around one anchor value, plus the controls to compare them with.

An anchor is one moral value (e.g. "Compassion"). From it we build, for each profile size n (3, 4, 5):

    similar   the anchor + its n-1 closest values from its own similarity-bank cluster
    random    the anchor + n-1 random values from OTHER clusters      (several independent draws)
    distant   the anchor + the n-1 values LEAST similar to it overall
and, at size 1, the anchor alone. "none" (no profile) is added by the experiment script.

Design choices (so that differences between conditions mean something):
  * Nested: a size-4 profile is the size-3 profile plus one value, for every kind. So a change across sizes comes
    from adding one more value, not from swapping values.
  * Deterministic: ties are broken alphabetically; random draws are seeded by (seed, anchor, draw).
  * Same weighting rule for every size and kind (default: anchor 0.5, the rest share the other 0.5 equally).
  * Small clusters: most clusters have 2-5 values, so a 5-value profile often cannot come from the cluster alone.
        scope="cluster"  strict: that size is skipped for that anchor (the experiment reports what was skipped)
        scope="tree"     fill the gap from the smallest group above the cluster in the dendrogram (closest first).
                         Cluster members always come first, and `n_outside` records how many values are not from
                         the anchor's own cluster, so these profiles can be reported separately.

Usage:
    from ethos.value_banks import SimilarityBank
    from ethos.multi_profiles import build_conditions
    conds = build_conditions(SimilarityBank.load(), ["Compassion", "Autonomy"], sizes=(3, 4, 5), n_random=3)
    conds["Compassion|similar|n3"]["profile"]   # [("Compassion", 0.5), ("Sensitivity", 0.25), ...]
"""
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ethos.llm.dmp import format_profile

KINDS = ("alone", "similar", "random", "distant")


# --- ranking helpers --------------------------------------------------------------------------------------------

def _sim(bank, a, b):
    return bank.similarity(a, b)


def _by_similarity(bank, anchor, values):
    """values sorted by similarity to the anchor, highest first; ties alphabetical (so reruns never change)."""
    return sorted(values, key=lambda v: (-_sim(bank, anchor, v), v))


def _tree_nodes(bank):
    """Every group of the average-linkage tree over the bank's values, as (frozenset of values) in merge order."""
    values = list(bank.matrix.index)
    dist = 1 - bank.matrix.loc[values, values].to_numpy()
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(np.clip((dist + dist.T) / 2, 0, None), checks=False), method="average")
    members = {i: frozenset([v]) for i, v in enumerate(values)}
    nodes = []
    for j, (a, b, _, _) in enumerate(z):
        members[len(values) + j] = members[int(a)] | members[int(b)]
        nodes.append(members[len(values) + j])
    return nodes


def similar_order(bank, anchor, scope="tree", need=4):
    """Other values in the order they would be added to the anchor: its cluster members by similarity, then (scope
    "tree" only) members of the smallest dendrogram group that holds at least `need` others, by similarity.
    Returns [(value, source)] with source "cluster" or "tree"."""
    own = [v for v in bank.members(bank.cluster_of(anchor)) if v != anchor]
    out = [(v, "cluster") for v in _by_similarity(bank, anchor, own)]
    if scope == "tree" and len(out) < need:
        for node in _tree_nodes(bank):
            if anchor in node and len(node) - 1 >= need:
                extra = [v for v in node if v != anchor and v not in own]
                out += [(v, "tree") for v in _by_similarity(bank, anchor, extra)]
                break
    return out


def random_order(bank, anchor, seed=0, draw=0):
    """A random order of the values outside the anchor's cluster (seeded; prefixes give nested profiles)."""
    inside = set(bank.members(bank.cluster_of(anchor)))
    pool = sorted(v for v in bank.matrix.index if v not in inside)
    key = [seed, sorted(bank.matrix.index).index(anchor), draw]
    return list(np.random.default_rng(key).permutation(pool))


def distant_order(bank, anchor):
    """All other values, least similar to the anchor first (ties alphabetical)."""
    others = [v for v in bank.matrix.index if v != anchor]
    return sorted(others, key=lambda v: (_sim(bank, anchor, v), v))


# --- weights and quality measures ------------------------------------------------------------------------------

def weights(n, scheme="anchor", anchor_weight=0.5):
    """n weights summing to 1; the anchor is first. "anchor": anchor_weight, the rest share the remainder equally.
    "equal": all the same. A single value always gets 1.0."""
    if n == 1:
        return [1.0]
    if scheme == "equal":
        return [1 / n] * n
    if scheme == "anchor":
        return [anchor_weight] + [(1 - anchor_weight) / (n - 1)] * (n - 1)
    raise ValueError(f"unknown weighting {scheme!r} (anchor, equal)")


def coherence(bank, values):
    """Mean pairwise similarity among a profile's values (how close the profile hangs together)."""
    pairs = [(a, b) for i, a in enumerate(values) for b in values[i + 1:]]
    return float(np.mean([_sim(bank, a, b) for a, b in pairs])) if pairs else float("nan")


def anchor_similarity(bank, anchor, values):
    """Mean similarity of the non-anchor values to the anchor."""
    rest = [v for v in values if v != anchor]
    return float(np.mean([_sim(bank, anchor, v) for v in rest])) if rest else float("nan")


# --- building conditions ---------------------------------------------------------------------------------------

def _entry(bank, anchor, kind, size, extras, scheme, scope="", draw=None, n_outside=0):
    values = [anchor] + list(extras)
    profile = list(zip(values, weights(len(values), scheme)))
    return {
        "anchor": anchor, "kind": kind, "size": len(values), "draw": draw, "scope": scope, "n_outside": n_outside,
        "profile": profile, "text": format_profile(profile),
        "coherence": coherence(bank, values), "anchor_similarity": anchor_similarity(bank, anchor, values),
        "n_clusters": len(bank.clusters_in(values)),
    }


def name_of(anchor, kind, size, draw=None):
    return f"{anchor}|{kind}|n{size}" + (f"|d{draw}" if draw is not None else "")


def build_conditions(bank, anchors, sizes=(3, 4, 5), n_random=3, scope="tree", scheme="anchor", seed=0,
                     kinds=("alone", "similar", "random", "distant")):
    """{condition name: entry} for every anchor, size and kind; plus the list of (anchor, size) that were skipped
    because scope="cluster" could not fill them. Returns (conditions, skipped)."""
    out, skipped = {}, []
    for anchor in anchors:
        if anchor not in bank.matrix.index:
            raise KeyError(f"{anchor!r} is not a value in the taxonomy")
        if "alone" in kinds:
            out[name_of(anchor, "alone", 1)] = _entry(bank, anchor, "alone", 1, [], scheme)
        order = similar_order(bank, anchor, scope, need=max(sizes) - 1)
        for n in sizes:
            if "similar" in kinds:
                if len(order) < n - 1:
                    skipped.append((anchor, n))
                else:
                    pick = order[: n - 1]
                    out[name_of(anchor, "similar", n)] = _entry(
                        bank, anchor, "similar", n, [v for v, _ in pick], scheme, scope,
                        n_outside=sum(s == "tree" for _, s in pick))
            if "random" in kinds:
                for d in range(n_random):
                    out[name_of(anchor, "random", n, d)] = _entry(
                        bank, anchor, "random", n, random_order(bank, anchor, seed, d)[: n - 1], scheme, "other",
                        draw=d, n_outside=n - 1)
            if "distant" in kinds:
                out[name_of(anchor, "distant", n)] = _entry(
                    bank, anchor, "distant", n, distant_order(bank, anchor)[: n - 1], scheme, "far", n_outside=n - 1)
    return out, skipped


def feasibility(bank, sizes=(3, 4, 5)):
    """{value: {size: True/False}}: can the anchor's own cluster alone fill a profile of that size?"""
    return {v: {n: len(bank.members(bank.cluster_of(v))) >= n for n in sizes} for v in bank.matrix.index}
