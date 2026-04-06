from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class LoadedModel:
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str


DTYPE_MAP = {
    "cuda": torch.float16,
    "cpu": torch.float32,
}


def load_causal_lm(model_name: str, device: str, trust_remote_code: bool = False) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = DTYPE_MAP.get(device, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return LoadedModel(name=model_name, model=model, tokenizer=tokenizer, device=device)
