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


def test_continuous_sessions_have_mixed_rounds_lengths_and_random_think_time():
    config = {
        "session_arrival_duration_s": 20,
        "session_rate": 8,
        "seed": 42,
        "length_variation": 0.2,
        "round_ratios": [2, 3, 5],
        "mean_new_tokens_per_round": [64, 32, 32],
        "mean_return_tokens_per_round": [8, 12, 16],
        "mean_think_time_ms": [0, 1000, 2000],
        "think_time_cv": 1.0,
    }
    rows = generate(config)
    assert rows == generate(config)
    first = [row for row in rows if row["metadata"]["turn"] == 0]
    assert all(0 < row["timestamp"] < 20000 for row in first)
    assert min(row["timestamp"] for row in first) < 1000
    assert max(row["timestamp"] for row in first) > 19000
    assert len({row["prompt_len"] for row in first}) > 10
    assert len({row["output_len"] for row in rows}) > 5
    sessions = {}
    for row in rows:
        sessions.setdefault(row["metadata"]["session_id"], []).append(row)
    assert {len(turns) for turns in sessions.values()} == {1, 2, 3}
    waits = []
    for turns in sessions.values():
        for before, after in zip(turns, turns[1:]):
            history = before["prompt"] + [1] * before["output_len"]
            assert after["prompt"][: len(history)] == history
            assert after["timestamp"] is None
            waits.append(after["metadata"]["think_time_us"])
    assert min(waits) > 0 and len(set(waits)) > 50
    changed_lengths = generate({**config, "mean_new_tokens_per_round": [80, 40, 40]})
    assert [r["timestamp"] for r in first] == [
        r["timestamp"] for r in changed_lengths if r["metadata"]["turn"] == 0
    ]
