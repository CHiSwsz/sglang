import pytest
from sglang_simulator.master_trace.arrivals import SessionTimeline


def plan():
    return [
        {"timestamp": 10, "metadata": {"session_id": 0, "turn": 0}},
        {
            "timestamp": None,
            "metadata": {
                "session_id": 0,
                "turn": 1,
                "arrival_mode": "after_completion",
                "think_time_us": 5000,
            },
        },
        {"timestamp": 20, "metadata": {"session_id": 1, "turn": 0}},
    ]


def test_slow_response_delays_only_its_followup_and_exports_absolute_arrivals():
    timeline = SessionTimeline(plan())
    assert [row["timestamp"] for _, row in timeline.initial] == [10, 20]
    first, second, other = timeline.requests
    timeline.complete(other, 25000)
    rid, row = timeline.complete(first, 90000)
    assert rid == second and row["timestamp"] == 95
    assert timeline.requests[other]["timestamp"] == 20
    timeline.complete(second, 110000)
    resolved = timeline.resolved_rows()
    assert [row["timestamp"] for row in resolved] == [10, 20, 95]
    # Export is regular AutoBench: loading it again must not add think time twice.
    fixed = SessionTimeline(resolved)
    assert len(fixed.initial) == 3 and not fixed.followers
    faster = SessionTimeline(plan())
    assert faster.complete(first, 30000)[1]["timestamp"] == 35


def test_unresolved_or_noncausal_sessions_fail():
    timeline = SessionTimeline(plan())
    with pytest.raises(RuntimeError, match="not all"):
        timeline.resolved_rows()
    bad = plan()
    bad[1]["metadata"]["turn"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        SessionTimeline(bad)
    timeline = SessionTimeline(
        [
            {"timestamp": 0, "metadata": {"session_id": 0, "turn": 0}},
            {"timestamp": 1, "metadata": {"session_id": 0, "turn": 1}},
        ]
    )
    first, second = timeline.requests
    timeline.complete(first, 2000)
    timeline.complete(second, 3000)
    with pytest.raises(RuntimeError, match="precedes"):
        timeline.validate()
