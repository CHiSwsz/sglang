"""Seeded session plans with bench_mix-style conversation history.

Each turn appends the preceding response and the new user input. The official
simulator samples token 1, so that exact response is used in following prompts.
Later turns wait for simulated completion plus sampled positive think time.
The cluster driver resolves the plan into timestamped AutoBench JSONL.
No tokenizer, model weights, GPU, HTTP server, or wall clock is required.
"""

import argparse
import hashlib
import json
import math
import random
from array import array
from pathlib import Path


def generate(config: dict, *, compact=False) -> list[dict]:
    rng = random.Random(config.get("seed", 42))
    sessions = config.get("sessions")
    duration = config.get("session_arrival_duration_s")
    rate = config["session_rate"]
    inputs = config["mean_new_tokens_per_round"]
    outputs = config["mean_return_tokens_per_round"]
    legacy = "mean_inter_round_interval_ms" in config
    if legacy and "mean_think_time_ms" in config:
        raise ValueError(
            "choose completion-relative think time or legacy fixed offsets"
        )
    intervals = config[
        "mean_inter_round_interval_ms" if legacy else "mean_think_time_ms"
    ]
    think_cv = config.get("think_time_cv", 1.0)
    weights = config.get("round_ratios", [0] * (len(inputs) - 1) + [1])
    if (sessions is None) == (duration is None):
        raise ValueError(
            "provide exactly one of sessions or session_arrival_duration_s"
        )
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("session_rate must be finite and positive")
    if sessions is not None and (not isinstance(sessions, int) or sessions <= 0):
        raise ValueError("sessions must be a positive integer")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("session_arrival_duration_s must be finite and positive")
    if not math.isfinite(think_cv) or think_cv < 0:
        raise ValueError("think_time_cv must be finite and nonnegative")
    if not inputs or not (
        len(inputs) == len(outputs) == len(intervals) == len(weights)
    ):
        raise ValueError("all per-round arrays must have the same nonzero length")
    if any(not math.isfinite(v) or v <= 0 for v in inputs + outputs) or any(
        not math.isfinite(v) or v < 0 for v in intervals + weights
    ):
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
    # New workloads use independent streams: changing prompt lengths or think
    # times must not change the externally offered new-session arrivals.
    arrival_rng = rng if legacy else random.Random(config.get("seed", 42) + 1)
    think_rng = random.Random(config.get("seed", 42) + 2)
    arrival_ms = 0.0
    session = 0
    while sessions is None or session < sessions:
        if session or not legacy:
            arrival_ms += arrival_rng.expovariate(rate) * 1000
        if duration is not None and arrival_ms >= duration * 1000:
            break
        rounds = rng.choices(range(1, len(inputs) + 1), weights=weights)[0]
        prompt = []
        timestamp = arrival_ms
        for turn in range(rounds):
            metadata = {"session_id": session, "turn": turn}
            if not legacy and turn:
                mean = intervals[turn]
                sigma = math.sqrt(math.log1p(think_cv**2))
                delay_ms = (
                    think_rng.lognormvariate(math.log(mean) - sigma**2 / 2, sigma)
                    if mean > 0
                    else 0
                )
                metadata.update(
                    arrival_mode="after_completion",
                    think_time_us=round(delay_ms * 1000),
                )
            prompt.extend(
                rng.randrange(16, vocab_size) for _ in range(length(inputs[turn]))
            )
            output_len = length(outputs[turn])
            rows.append(
                {
                    "prompt": array("I", prompt) if compact else list(prompt),
                    "prompt_len": len(prompt),
                    "output_len": output_len,
                    "timestamp": round(timestamp, 6) if legacy or turn == 0 else None,
                    "metadata": metadata,
                }
            )
            prompt.extend([1] * output_len)
            if legacy and turn + 1 < rounds:
                # Open-loop requested arrival. The cluster driver must check
                # session causality against actual simulated completion times.
                timestamp += intervals[turn + 1]
        session += 1
    if not rows:
        raise ValueError("no sessions arrived within the configured horizon")
    if legacy:
        rows.sort(key=lambda row: (row["timestamp"], row["metadata"]["session_id"]))
    return rows


def write_workload(config_path: Path, output: Path):
    config = json.loads(config_path.read_text())
    rows = generate(config, compact=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), default=list) + "\n")
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "format": "autobench"
        if "mean_inter_round_interval_ms" in config
        else "session_plan",
        "time_unit": "ms",
        "requests": len(rows),
        "config": config,
        "sha256": digest,
        "response_tokens": "token 1, matching the official CPU simulator sampler",
        "arrival_mode": (
            "legacy fixed offsets; validate causality"
            if "mean_inter_round_interval_ms" in config
            else "Poisson new sessions; later turns follow completion plus lognormal think time"
        ),
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
