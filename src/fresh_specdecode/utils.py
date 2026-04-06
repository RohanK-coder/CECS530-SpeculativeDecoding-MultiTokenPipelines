import json
import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(user_choice: str) -> str:
    if user_choice != "auto":
        return user_choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt:
        return prompt
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --prompt or --prompt-file.")


def pretty_json(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
