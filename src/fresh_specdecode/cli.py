from __future__ import annotations

import argparse

from fresh_specdecode.config import GenerationConfig
from fresh_specdecode.decoders import BaselineDecoder, PipelinedSpeculativeDecoder, SpeculativeDecoder
from fresh_specdecode.models import load_causal_lm
from fresh_specdecode.utils import choose_device, pretty_json, read_prompt, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline, speculative, and pipelined decoding demo")
    parser.add_argument("--mode", choices=["baseline", "speculative", "pipeline"], required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--target-model", type=str, default="distilgpt2")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--speculation-k", type=int, default=4)
    parser.add_argument("--pipeline-window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = GenerationConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        max_new_tokens=args.max_new_tokens,
        speculation_k=args.speculation_k,
        pipeline_window=args.pipeline_window,
        seed=args.seed,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    prompt = read_prompt(args.prompt, args.prompt_file)
    set_seed(config.seed)
    device = choose_device(config.device)

    target = load_causal_lm(config.target_model, device=device, trust_remote_code=config.trust_remote_code)

    if args.mode == "baseline":
        decoder = BaselineDecoder(target)
    else:
        draft = load_causal_lm(config.draft_model, device=device, trust_remote_code=config.trust_remote_code)
        if args.mode == "speculative":
            decoder = SpeculativeDecoder(draft, target, speculation_k=config.speculation_k)
        else:
            decoder = PipelinedSpeculativeDecoder(
                draft,
                target,
                speculation_k=config.speculation_k,
                pipeline_window=config.pipeline_window,
            )

    text, metrics = decoder.generate(prompt=prompt, max_new_tokens=config.max_new_tokens)
    print("\n=== GENERATED TEXT ===\n")
    print(text)
    print("\n=== METRICS ===\n")
    print(pretty_json(metrics.to_dict()))


if __name__ == "__main__":
    main()
