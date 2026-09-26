from sglang_simulator.master_trace.storage import MHALayout, SharedStorage
from sglang_simulator.master_trace.trace import TraceRecorder


def test_cross_instance_reuse_waits_for_commit_and_evicts():
    layout = MHALayout("Qwen/Qwen3-8B", 36, 8, 128, 2, 256, 8)
    recorder = TraceRecorder({})
    storage = SharedStorage(recorder, layout, 16 * layout.object_bytes, 1 << 30)
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
