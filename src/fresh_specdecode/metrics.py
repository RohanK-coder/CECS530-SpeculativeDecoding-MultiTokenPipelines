from dataclasses import asdict, dataclass


@dataclass(slots=True)
class DecodeMetrics:
    mode: str
    generated_tokens: int
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    draft_rounds: int = 0
    verify_rounds: int = 0
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_draft_tokens + self.rejected_draft_tokens
        return 0.0 if total == 0 else self.accepted_draft_tokens / total

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["acceptance_rate"] = round(self.acceptance_rate, 4)
        return payload
