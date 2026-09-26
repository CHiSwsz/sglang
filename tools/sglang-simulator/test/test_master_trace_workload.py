from sglang_simulator.master_trace.workload import generate


def test_multiturn_history_and_arrivals():
    config = {
        "sessions": 12,
        "session_rate": 4,
        "seed": 73,
        "length_variation": 0,
        "mean_new_tokens_per_round": [128, 32, 64],
        "mean_return_tokens_per_round": [8, 16, 24],
        "mean_inter_round_interval_ms": [0, 1000, 2000],
    }
    rows = generate(config)
    assert rows == generate(config)
    assert len(rows) == 36
    assert [r["timestamp"] for r in rows] == sorted(r["timestamp"] for r in rows)
    for session in range(12):
        turns = [r for r in rows if r["metadata"]["session_id"] == session]
        for before, after in zip(turns, turns[1:]):
            history = before["prompt"] + [1] * before["output_len"]
            assert after["prompt"][: len(history)] == history
            assert after["timestamp"] > before["timestamp"]
    assert rows[0]["prompt"] != rows[1]["prompt"]
