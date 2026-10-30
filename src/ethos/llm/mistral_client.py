"""Thin wrapper to reach a local Mistral model through Ollama's OpenAI-compatible endpoint.

Setup: `ollama pull ministral-<model-version>` (Ollama must be running). The model and
server are chosen by the caller, e.g. from configs/generation.yaml.
"""
import json

from openai import APIConnectionError, OpenAI

DEFAULT_BASE_URL = "http://localhost:11434/v1"


def get_client(model, base_url=DEFAULT_BASE_URL):
    """Return an OpenAI client for the local Ollama server, after checking `model` is pulled."""
    client = OpenAI(base_url=base_url, api_key="ollama")  # Ollama ignores the key
    try:
        available = sorted(m.id for m in client.models.list())
    except APIConnectionError as e:
        raise RuntimeError(f"Can't reach Ollama at {base_url}. Is it running? (`ollama serve`)") from e
    if model not in available:
        raise ValueError(
            f"Model '{model}' is not available in Ollama. Pull it with `ollama pull {model}`. "
            f"Available: {', '.join(available) or 'none'}"
        )
    return client


def chat_json(client, model, system, user, schema, temperature=0.5, max_tokens=600):
    """One chat call that must return JSON matching `schema`. Returns a dict."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": schema, "strict": True},
        },
    )
    return json.loads(resp.choices[0].message.content)
