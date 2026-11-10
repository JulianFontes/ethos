"""Project paths, so scripts work from any folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN_SAMPLE = ROOT / "data/processed/consensus_n125_train"   # tuning and experiments
TEST_SAMPLE = ROOT / "data/processed/consensus_n125_test"     # held-out evaluation (Detoxify comparison)
EXPERIMENTS = ROOT / "data/experiments"
