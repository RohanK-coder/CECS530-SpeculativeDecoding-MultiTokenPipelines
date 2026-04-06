from dataclasses import dataclass


@dataclass(slots=True)
class GenerationConfig:
    draft_model: str = "sshleifer/tiny-gpt2"
    target_model: str = "distilgpt2"
    max_new_tokens: int = 64
    speculation_k: int = 4
    pipeline_window: int = 3
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 7
    device: str = "auto"
    trust_remote_code: bool = False
