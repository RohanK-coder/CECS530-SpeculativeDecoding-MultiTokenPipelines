from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from fresh_specdecode.metrics import DecodeMetrics


@dataclass(slots=True)
class VerificationResult:
    accepted_prefix_len: int
    fallback_token: int | None
    accepted_tokens: list[int]
    rejected_count: int


class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, speculation_k: int = 4):
        if draft_model.device != target_model.device:
            raise ValueError("Draft and target models must be on the same device.")
        self.draft = draft_model
        self.target = target_model
        self.speculation_k = speculation_k

    @staticmethod
    def verify_proposed_tokens(target_argmax: list[int], proposed: list[int]) -> VerificationResult:
        accepted = []
        for idx, token in enumerate(proposed):
            if idx >= len(target_argmax):
                break
            if target_argmax[idx] == token:
                accepted.append(token)
                continue
            return VerificationResult(
                accepted_prefix_len=len(accepted),
                fallback_token=target_argmax[idx],
                accepted_tokens=accepted,
                rejected_count=len(proposed) - len(accepted),
            )
        return VerificationResult(
            accepted_prefix_len=len(accepted),
            fallback_token=None,
            accepted_tokens=accepted,
            rejected_count=len(proposed) - len(accepted),
        )

    @torch.no_grad()
    def _draft_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, count: int) -> list[int]:
        drafted: list[int] = []
        running_ids = input_ids
        running_mask = attention_mask

        for _ in range(count):
            outputs = self.draft.model(input_ids=running_ids, attention_mask=running_mask)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            drafted.append(int(next_token.item()))
            running_ids = torch.cat([running_ids, next_token], dim=1)
            running_mask = torch.cat(
                [running_mask, torch.ones((1, 1), device=running_mask.device, dtype=running_mask.dtype)],
                dim=1,
            )
            if next_token.item() == self.target.tokenizer.eos_token_id:
                break
        return drafted

    @torch.no_grad()
    def _target_argmax_for_block(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        proposed: list[int],
    ) -> list[int]:
        block = torch.tensor([proposed], device=input_ids.device, dtype=input_ids.dtype)
        test_ids = torch.cat([input_ids, block], dim=1)
        test_mask = torch.cat(
            [attention_mask, torch.ones((1, len(proposed)), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        outputs = self.target.model(input_ids=test_ids, attention_mask=test_mask)

        prefix_len = input_ids.shape[1]
        logits = outputs.logits[0]
        argmax_tokens: list[int] = []
        for i in range(len(proposed)):
            prediction_position = prefix_len - 1 + i
            token_id = int(logits[prediction_position].argmax(dim=-1).item())
            argmax_tokens.append(token_id)
        return argmax_tokens

    @torch.no_grad()
    def _target_next_after_prefix(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
        outputs = self.target.model(input_ids=input_ids, attention_mask=attention_mask)
        return int(outputs.logits[:, -1, :].argmax(dim=-1).item())

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, DecodeMetrics]:
        tokenizer = self.target.tokenizer
        device = self.target.device
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        accepted_total = 0
        rejected_total = 0
        draft_rounds = 0
        verify_rounds = 0

        start = time.perf_counter()
        while (input_ids.shape[1] - prompt_len) < max_new_tokens:
            remaining = max_new_tokens - (input_ids.shape[1] - prompt_len)
            proposed = self._draft_tokens(input_ids, attention_mask, min(self.speculation_k, remaining))
            if not proposed:
                break
            draft_rounds += 1
            verify_rounds += 1

            target_argmax = self._target_argmax_for_block(input_ids, attention_mask, proposed)
            result = self.verify_proposed_tokens(target_argmax, proposed)

            emitted = result.accepted_tokens.copy()
            if result.fallback_token is None and len(emitted) == len(proposed):
                if (input_ids.shape[1] - prompt_len + len(emitted)) < max_new_tokens:
                    extended_ids = torch.tensor([emitted], device=device, dtype=input_ids.dtype)
                    extended_input_ids = torch.cat([input_ids, extended_ids], dim=1)
                    extended_mask = torch.cat(
                        [attention_mask, torch.ones((1, len(emitted)), device=device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                    emitted.append(self._target_next_after_prefix(extended_input_ids, extended_mask))
            elif result.fallback_token is not None:
                emitted.append(result.fallback_token)

            if not emitted:
                break

            accepted_total += result.accepted_prefix_len
            rejected_total += result.rejected_count

            emit_tensor = torch.tensor([emitted], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, emit_tensor], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, len(emitted)), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )

            if emitted[-1] == tokenizer.eos_token_id:
                break

        elapsed = time.perf_counter() - start
        generated_ids = input_ids[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_tokens = generated_ids.numel() - prompt_len
        metrics = DecodeMetrics(
            mode="speculative",
            generated_tokens=new_tokens,
            accepted_draft_tokens=accepted_total,
            rejected_draft_tokens=rejected_total,
            draft_rounds=draft_rounds,
            verify_rounds=verify_rounds,
            elapsed_seconds=elapsed,
            tokens_per_second=0.0 if elapsed == 0 else new_tokens / elapsed,
        )
        return generated_text, metrics
