"""Kaleido (Value Kaleidoscope, Sorensen et al. 2024): values, rights and duties relevant to a text.

Setup: the models are gated. Accept the license on huggingface.co/allenai/kaleido-large and set HF_TOKEN in .env.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Kaleido:
    def __init__(self, model_name):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.tasks = self.model.config.task_specific_params   # prompt template and labels per task

    def _encode(self, prompts):
        return self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

    @torch.no_grad()
    def generate(self, text, n=10):
        """Up to n distinct values/rights/duties, e.g. ["Value: Respect", "Right: Right to free speech", ...]."""
        prompt = self.tasks["generate"]["template"].replace("ACTION", text)
        out = self.model.generate(**self._encode([prompt]), max_length=32, num_beams=n, num_return_sequences=n)
        return list(dict.fromkeys(self.tokenizer.batch_decode(out, skip_special_tokens=True)))

    @torch.no_grad()
    def score(self, task, text, items):
        """Label probabilities per item: task "relevance" -> Yes/No, "valence" -> Supports/Opposes/Either."""
        template, labels = self.tasks[task]["template"], self.tasks[task]["label_to_id"]
        prompts = []
        for item in items:
            kind, _, name = item.partition(": ")   # "Value: Respect" -> "Value", "Respect"
            prompts.append(template.replace("ACTION", text).replace("VRD", kind).replace("TEXT", name))
        start = torch.tensor([self.tasks[task]["dummy_decoder_ids"]] * len(items), device=self.model.device)
        logits = self.model(**self._encode(prompts), decoder_input_ids=start).logits[:, 0, list(labels.values())]
        return [dict(zip(labels, p)) for p in logits.softmax(-1).tolist()]
