"""Project paths, so scripts work from any folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMMENTS = ROOT / "data/processed/civil-comments/comments"                    # every filtered comment (train + test)
EMBEDDINGS = COMMENTS / "embeddings"                            # embeddings.parquet and checks of it
TRAIN_SAMPLE = ROOT / "data/processed/civil-comments/consensus_n125_train"   # tuning and experiments
SINGLE_VALUES = TRAIN_SAMPLE / "single_values"                   # ratings.csv (raw LLM answers) + scores.csv (LLM vs human toxicity per comment x value): policy training data
TEST_SAMPLE = ROOT / "data/processed/civil-comments/consensus_n125_test"     # held-out evaluation (Detoxify comparison)
EXPERIMENTS = ROOT / "data/experiments/civil-comments"
