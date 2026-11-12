"""Steering an LLM judge's TOXIC / NOT TOXIC answers, two ways that can be combined:

prompt      prompt steering: a value or moral profile in the system prompt (conditions, system prompt, names)
activation  activation steering with HF transformers: PCA value vectors (VISPA's recipe) and adding them to layers
in_vllm     activation steering in vLLM: steering settings for a run, the steered server, capturing hidden states
pairs       VISPA contrastive pair files and layer ranges

Import the submodules directly (e.g. `from ethos.steering.prompt import system_prompt`): in_vllm and pairs also run
in vLLM's environment, which lacks pandas and scikit-learn, so this file imports nothing.
"""
