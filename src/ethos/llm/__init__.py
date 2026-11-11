"""LLM backends. connect(model) picks one from the model name:

  vllm:<hf model id>   a vLLM server, e.g. vllm:Qwen/Qwen3-8B (VLLM_BASE_URL, default http://localhost:8000/v1)
  <name>:<tag>         local Ollama, e.g. qwen3:8b, gpt-oss:20b (Ollama names always carry a tag)
  gpt-...              OpenAI API (needs OPENAI_API_KEY), e.g. gpt-6-luna
"""


def connect(model):
    """Return (backend module, client, model name as the backend knows it). The module has chat_json()."""
    from ethos.llm import ollama_client, openai_client, vllm_client
    if model.startswith("vllm:"):
        name = model.removeprefix("vllm:")
        return vllm_client, vllm_client.get_client(name), name
    if ":" in model:
        return ollama_client, ollama_client.get_client(model), model
    if model.startswith("gpt"):
        return openai_client, openai_client.get_client(model), model
    raise ValueError(f"Unknown backend for '{model}': use gpt-..., vllm:<hf id> or an Ollama name:tag")
