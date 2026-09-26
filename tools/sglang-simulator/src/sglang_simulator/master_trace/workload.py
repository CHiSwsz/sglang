"""Deterministic token-level version of HiCache bench_mix's session generator.

Each turn appends the preceding response and the new user input. The official
simulator samples token 1, so that exact response is used in following prompts.
Arrival timestamps are milliseconds, as required by the AutoBench contract.
No tokenizer, model weights, GPU, HTTP server, or wall clock is required.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def generate(config: dict) -> list[dict]:
    rng = random.Random(config.get("seed", 42))
    sessions = config["sessions"]
    rate = config["session_rate"]
    inputs = config["mean_new_tokens_per_round"]
    outputs = config["mean_return_tokens_per_round"]
    intervals = config["mean_inter_round_interval_ms"]
    weights = config.get("round_ratios", [0] * (len(inputs) - 1) + [1])
    if sessions <= 0 or rate <= 0:
        raise ValueError("sessions and session_rate must be positive")
    if not inputs or not (
        len(inputs) == len(outputs) == len(intervals) == len(weights)
    ):
        raise ValueError("all per-round arrays must have the same nonzero length")
    if any(v <= 0 for v in inputs + outputs) or any(v < 0 for v in intervals + weights):
        raise ValueError("invalid lengths, intervals or round weights")
    if sum(weights) <= 0:
        raise ValueError("round_ratios must contain a positive weight")
    variation = config.get("length_variation", 0.2)
    if not 0 <= variation < 1:
        raise ValueError("length_variation must be in [0, 1)")
    vocab_size = config.get("vocab_size", 151936)
    if vocab_size <= 16:
        raise ValueError("vocab_size must exceed 16")

    def length(mean):
        return max(
            1, round(rng.uniform(mean * (1 - variation), mean * (1 + variation)))
        )

    rows = []
    arrival_ms = 0.0
    for session in range(sessions):
        if session:
            arrival_ms += rng.expovariate(rate) * 1000
        rounds = rng.choices(range(1, len(inputs) + 1), weights=weights)[0]
        prompt = []
        timestamp = arrival_ms
        for turn in range(rounds):
            prompt.extend(
                rng.randrange(16, vocab_size) for _ in range(length(inputs[turn]))
            )
            output_len = length(outputs[turn])
            rows.append(
                {
                    "prompt": list(prompt),
                    "prompt_len": len(prompt),
                    "output_len": output_len,
                    "timestamp": round(timestamp, 6),
                    "metadata": {"session_id": session, "turn": turn},
                }
            )
            prompt.extend([1] * output_len)
            if turn + 1 < rounds:
                # Open-loop requested arrival. The cluster driver must check
                # session causality against actual simulated completion times.
                timestamp += intervals[turn + 1]
    return sorted(
        rows, key=lambda row: (row["timestamp"], row["metadata"]["session_id"])
    )


def write_workload(config_path: Path, output: Path):
    config = json.loads(config_path.read_text())
    rows = generate(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "format": "autobench",
        "time_unit": "ms",
        "requests": len(rows),
        "config": config,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "response_tokens": "token 1, matching the official CPU simulator sampler",
        "arrival_mode": "open_loop; validate each turn starts after preceding completion",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_workload(args.config, args.output)


if __name__ == "__main__":
    main()
