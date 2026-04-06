from __future__ import annotations

import time

import torch

from fresh_specdecode.metrics import DecodeMetrics


class BaselineDecoder:
    def __init__(self, target_model):
        self.target = target_model

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, DecodeMetrics]:
        tokenizer = self.target.tokenizer
        model = self.target.model
        device = self.target.device

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        start = time.perf_counter()
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            if next_token.item() == tokenizer.eos_token_id:
                break

        elapsed = time.perf_counter() - start
        generated_ids = input_ids[0]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_tokens = generated_ids.numel() - encoded["input_ids"].shape[1]
        metrics = DecodeMetrics(
            mode="baseline",
            generated_tokens=new_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=0.0 if elapsed == 0 else new_tokens / elapsed,
        )
        return text, metrics
