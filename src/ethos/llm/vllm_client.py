"""Thin wrapper to reach a model served by vLLM (OpenAI-compatible server), with the same interface as ollama_client.

Setup: a running `vllm serve <model>` (scripts/shell/slurm_vllm.sh starts one on a cluster GPU). The server URL is
VLLM_BASE_URL (default http://localhost:8000/v1).
"""
import os

from openai import APIConnectionError, OpenAI

from ethos.llm import ollama_client

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def get_client(model, base_url=None):
    """Return an OpenAI client for the vLLM server, after checking it serves `model`."""
    base_url = base_url or os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL)
    client = OpenAI(base_url=base_url, api_key="vllm")  # vLLM ignores the key unless started with --api-key
    try:
        served = sorted(m.id for m in client.models.list())
    except APIConnectionError as e:
        raise RuntimeError(f"Can't reach vLLM at {base_url}. Is `vllm serve` running?") from e
    if model not in served:
        raise ValueError(f"vLLM at {base_url} serves {served}, not '{model}'.")
    return client


def chat_json(client, model, system, user, schema, temperature=None, max_tokens=None, reasoning_effort=None):
    """Like ollama_client.chat_json. vLLM has no reasoning_effort; "none" switches thinking off through the chat
    template instead (Qwen3's enable_thinking)."""
    if reasoning_effort == "none":
        return ollama_client.chat_json(client, model, system, user, schema, temperature, max_tokens,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return ollama_client.chat_json(client, model, system, user, schema, temperature, max_tokens)
