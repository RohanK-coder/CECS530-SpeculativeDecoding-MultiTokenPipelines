#!/usr/bin/env bash
set -euo pipefail

python -m fresh_specdecode.cli \
  --mode speculative \
  --prompt-file prompts/sample.txt \
  --draft-model sshleifer/tiny-gpt2 \
  --target-model distilgpt2 \
  --speculation-k 4 \
  --max-new-tokens 64
