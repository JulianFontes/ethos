"""Thin wrapper to reach OpenAI models through the OpenAI API, with the same interface as ollama_client.

Setup: set OPENAI_API_KEY (e.g. in .env). The model is chosen in configs/judges.yaml.
"""
import os

from openai import OpenAI

from ethos.llm.ollama_client import chat_json  # same OpenAI SDK call; re-exported

__all__ = ["chat_json", "get_client"]


def get_client(model, base_url=None):
    """Return an OpenAI client. `base_url` is unused (kept to match ollama_client.get_client)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"Set OPENAI_API_KEY (e.g. in .env) to use {model}.")
    return OpenAI()
