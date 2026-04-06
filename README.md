# Speculative Decoding with MultiToken Pipelines

A standalone, from-scratch local project that demonstrates three text generation modes for causal language models:

1. **Baseline greedy decoding**
2. **Speculative decoding** using a draft and target model
3. **A simple multi-token pipeline** that overlaps drafting and verification in chunks

This project is intentionally self-contained and written to be easy to read and run locally. It does **not** reuse code from the repository you linked.

## Features

- Clean Python package layout
- CLI for all modes
- Hugging Face model loading
- GPU support if available, CPU fallback otherwise
- JSON metrics output
- Unit tests for acceptance/verification logic

## Recommended environment

- Python 3.10+
- 8GB+ RAM for tiny models
- Optional CUDA GPU for faster execution

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Run a baseline generation:

```bash
python -m fresh_specdecode.cli \
  --mode baseline \
  --prompt "Explain speculative decoding in plain English." \
  --target-model distilgpt2 \
  --max-new-tokens 64
```

Run speculative decoding:

```bash
python -m fresh_specdecode.cli \
  --mode speculative \
  --prompt "Explain speculative decoding in plain English." \
  --draft-model sshleifer/tiny-gpt2 \
  --target-model distilgpt2 \
  --speculation-k 4 \
  --max-new-tokens 64
```

Run the pipeline variant:

```bash
python -m fresh_specdecode.cli \
  --mode pipeline \
  --prompt "Write a short paragraph about efficient inference." \
  --draft-model sshleifer/tiny-gpt2 \
  --target-model distilgpt2 \
  --speculation-k 4 \
  --pipeline-window 3 \
  --max-new-tokens 64
```

## Notes on model choices

Default small-model pairing:
- Draft: `sshleifer/tiny-gpt2`
- Target: `distilgpt2`

These are chosen for easy local execution. You can swap in larger causal LM checkpoints if your hardware supports them.

## How the algorithms work

### Baseline
Greedy next-token decoding with the target model only.

### Speculative decoding
The draft model proposes up to `k` tokens. The target model evaluates the prompt plus the proposed block in one forward pass. Tokens are accepted left-to-right while the target model agrees with the draft. On the first disagreement, the target model's own preferred token is emitted and the draft remainder is discarded.

### Pipeline mode
This version keeps a short queue of drafted chunks and verifies them in FIFO order. It is a practical teaching implementation rather than a deeply optimized systems pipeline. The point is to show how chunked drafting and chunked verification can be organized in a reusable project.

## Output

The CLI prints:
- generated text
- structured metrics as JSON

Example metrics:

```json
{
  "mode": "speculative",
  "generated_tokens": 64,
  "accepted_draft_tokens": 31,
  "rejected_draft_tokens": 12,
  "acceptance_rate": 0.7209,
  "elapsed_seconds": 2.41,
  "tokens_per_second": 26.55
}
```

## Tests

```bash
pytest -q
```

## Project structure

```text
fresh-specdecode/
├── pyproject.toml
├── README.md
├── prompts/
│   └── sample.txt
├── scripts/
│   └── run_demo.sh
├── src/
│   └── fresh_specdecode/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── metrics.py
│       ├── models.py
│       ├── utils.py
│       └── decoders/
│           ├── __init__.py
│           ├── baseline.py
│           ├── pipeline.py
│           └── speculative.py
└── tests/
    └── test_speculative_helpers.py
```

## Troubleshooting

If a model download fails, make sure you have internet access on the machine where you run the project. If you are on CPU only, the first run may be slow.

If you want to avoid huge memory use, stick to:
- `sshleifer/tiny-gpt2`
- `distilgpt2`

## License

MIT
# CECS530-SpeculativeDecoding-MultiTokenPipelines
