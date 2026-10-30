"""Thin wrapper to reach any local model through Ollama's OpenAI-compatible endpoint.

Setup: `ollama pull <model>` (Ollama must be running). The model and server are chosen
by the caller, e.g. from configs/generation.yaml or configs/judges.yaml.
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


def chat_json(client, model, system, user, schema, temperature=None, max_tokens=None,
              reasoning_effort=None):
    """One chat call that must return JSON matching `schema`. Returns a dict.

    temperature / max_tokens / reasoning_effort are only sent when given; otherwise the
    model's defaults apply. reasoning_effort="none" turns off thinking for models like Qwen3
    (Ollama's OpenAI endpoint ignores `think`, but honors this).
    """
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": schema, "strict": True},
        },
        **kwargs,
    )
    return json.loads(resp.choices[0].message.content)
