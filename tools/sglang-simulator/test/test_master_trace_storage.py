import pytest
from sglang_simulator.master_trace.storage import MHALayout, SharedStorage
from sglang_simulator.master_trace.trace import TraceRecorder


def test_rejects_layout_that_would_require_multiple_mooncake_slices():
    with pytest.raises(ValueError, match="exceeds one Mooncake slice"):
        MHALayout("Qwen/Qwen3-8B", 36, 8, 128, 2, 256, 1)


@pytest.mark.parametrize("eviction_mode", ["record_remove", "master"])
def test_cross_instance_reuse_waits_for_commit_and_evicts(eviction_mode):
    layout = MHALayout("Qwen/Qwen3-8B", 36, 8, 128, 2, 256, 8)
    recorder = TraceRecorder({})
    storage = SharedStorage(
        recorder,
        layout,
        16 * layout.object_bytes,
        1 << 30,
        eviction_mode=eviction_mode,
    )
    complete = storage.write(0, ["prefix"])
    assert storage.lookup(1, ["prefix"]) == 0
    assert storage.stats["pending_keys"] == 16
    storage.advance(complete)
    assert storage.lookup(1, ["prefix"]) == 1
    loaded = storage.read(1, ["prefix"])
    storage.advance(loaded)
    complete = storage.write(0, ["replacement"])
    storage.advance(complete)
    assert storage.lookup(1, ["prefix"]) == 0
    assert storage.lookup(1, ["replacement"]) == 1
    assert storage.stats["evicted_keys"] == 16
    assert storage.used_bytes == storage.capacity
    assert layout.object_bytes == 2359296
    assert layout.keys(["abc"], 3) == ["Qwen-Qwen3-8B_abc_3_k", "Qwen-Qwen3-8B_abc_3_v"]
    removals = [event for event in recorder.events if event["op"] == "BatchRemove"]
    assert bool(removals) == (eviction_mode == "record_remove")
    starts = [event for event in recorder.events if event["op"] == "BatchPutStart"]
    assert all(event["value_sizes"] == [layout.object_bytes] * 2 for event in starts)


def test_large_gqa_layout_preserves_physical_kv_bytes():
    layout = MHALayout("Qwen/Qwen2.5-72B-Instruct", 80, 8, 128, 2, 128, 8)
    assert layout.object_bytes == 2621440
    # An entire 4K-token prompt across all eight TP ranks, K and V.
    assert (
        4096 // layout.page_size
    ) * 2 * layout.tp_size * layout.object_bytes == 1342177280


def test_replicated_kv_heads_and_multi_slice_state():
    layout = MHALayout("Qwen/Qwen3.8-27B", 16, 4, 256, 2, 256, 8)
    assert layout.object_bytes == 2 << 20
    assert (4096 // 256) * 2 * 8 * layout.object_bytes == 512 << 20
    recorder = TraceRecorder({})
    storage = SharedStorage(recorder, layout, 1 << 30, 1 << 30)
    keys = ["checkpoint_temporal", "checkpoint_conv_0"]
    complete = storage.write_keys("rank0", keys, [18 << 20, 368640])
    start = next(e for e in recorder.events if e["op"] == "BatchPutStart")
    assert start["value_slices"] == [[4194288] * 4 + [2097216], [368640]]
    assert storage.lookup_keys("rank1", keys) == [0, 0]
    storage.advance(complete)
    assert storage.lookup_keys("rank1", keys) == [1, 1]
    complete, found = storage.read_keys("rank1", keys)
    assert found == [True, True]
    assert all(storage.entries[key].pins == 1 for key in keys)
    storage.advance(complete)
    assert storage.stats["read_keys"] == 2
    assert storage.used_bytes == (18 << 20) + 368640


def test_eviction_between_lookup_and_read_returns_a_miss():
    layout = MHALayout("Qwen/Qwen3.8-27B", 16, 4, 256, 2, 256, 8)
    storage = SharedStorage(
        TraceRecorder({}), layout, 32 << 20, 1 << 30, eviction_mode="master"
    )
    storage.advance(storage.write(0, ["old"]))
    assert storage.lookup(1, ["old"]) == 1
    storage.advance(storage.write(0, ["new"]))
    done, hits = storage.read(1, ["old", "new"], return_hits=True)
    assert hits == [False, True]
    storage.advance(done)
    assert storage.stats["read_keys"] == 16
