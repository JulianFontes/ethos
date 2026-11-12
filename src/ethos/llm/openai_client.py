"""Thin wrapper to reach OpenAI models through the OpenAI API, with the same interface as ollama_client.

Setup: set OPENAI_API_KEY (e.g. in .env). The model is chosen in configs/judges/judges.yaml.
"""
import os
from pathlib import Path

from openai import OpenAI

from ethos.llm.ollama_client import chat_json  # same OpenAI SDK call; re-exported

__all__ = ["chat_json", "get_client"]


ROOT = Path(__file__).resolve().parents[3]


def load_env(path=ROOT / ".env"):
    """Put KEY=value lines from the project's .env into the environment (existing variables win)."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#") and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


def get_client(model, base_url=None):
    """Return an OpenAI client, reading OPENAI_API_KEY from .env if needed. `base_url` is unused (kept to match
    ollama_client.get_client)."""
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"Set OPENAI_API_KEY (e.g. in .env) to use {model}.")
    return OpenAI()
