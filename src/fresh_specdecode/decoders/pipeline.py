from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from fresh_specdecode.decoders.speculative import SpeculativeDecoder
from fresh_specdecode.metrics import DecodeMetrics


@dataclass(slots=True)
class DraftChunk:
    tokens: list[int]


class PipelinedSpeculativeDecoder(SpeculativeDecoder):
    def __init__(self, draft_model, target_model, speculation_k: int = 4, pipeline_window: int = 3):
        super().__init__(draft_model=draft_model, target_model=target_model, speculation_k=speculation_k)
        self.pipeline_window = max(1, pipeline_window)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, DecodeMetrics]:
        tokenizer = self.target.tokenizer
        device = self.target.device
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        queue: deque[DraftChunk] = deque()
        accepted_total = 0
        rejected_total = 0
        draft_rounds = 0
        verify_rounds = 0

        import time
        wall_start = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        if start_event is not None:
            start_event.record()

        while (input_ids.shape[1] - prompt_len) < max_new_tokens:
            while len(queue) < self.pipeline_window and (input_ids.shape[1] - prompt_len) < max_new_tokens:
                remaining = max_new_tokens - (input_ids.shape[1] - prompt_len)
                drafted = self._draft_tokens(input_ids, attention_mask, min(self.speculation_k, remaining))
                if not drafted:
                    break
                queue.append(DraftChunk(tokens=drafted))
                draft_rounds += 1
                if drafted[-1] == tokenizer.eos_token_id:
                    break

            if not queue:
                break

            chunk = queue.popleft()
            verify_rounds += 1
            target_argmax = self._target_argmax_for_block(input_ids, attention_mask, chunk.tokens)
            result = self.verify_proposed_tokens(target_argmax, chunk.tokens)

            emitted = result.accepted_tokens.copy()
            if result.fallback_token is None and len(emitted) == len(chunk.tokens):
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

            queue.clear()

            if emitted[-1] == tokenizer.eos_token_id:
                break

        elapsed = 0.0
        if start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            elapsed = max(1e-9, time.perf_counter() - wall_start)

        generated_ids = input_ids[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_tokens = generated_ids.numel() - prompt_len
        metrics = DecodeMetrics(
            mode="pipeline",
            generated_tokens=new_tokens,
            accepted_draft_tokens=accepted_total,
            rejected_draft_tokens=rejected_total,
            draft_rounds=draft_rounds,
            verify_rounds=verify_rounds,
            elapsed_seconds=elapsed,
            tokens_per_second=0.0 if elapsed == 0 else new_tokens / elapsed,
        )
        return generated_text, metrics
