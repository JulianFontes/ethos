"""VISPA contrastive pair files and layer ranges. Standard library only (also used in vLLM's environment)."""
import csv

def load_pairs(path):
    """(for, against) sentences from a VISPA pair file (<value>_context_controlled.csv: question_1, question_2)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["question_1"] for r in rows], [r["question_2"] for r in rows]

def parse_layers(spec):
    """"7-19" -> [7, ..., 19]; "3,5,9" -> [3, 5, 9]."""
    if "-" in spec:
        start, end = map(int, spec.split("-"))
        return list(range(start, end + 1))
    return [int(x) for x in spec.split(",")]
