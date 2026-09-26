import json

from sglang_simulator.master_trace.trace import TraceRecorder


def test_key_dependencies_preserve_visibility_without_serializing_reads():
    recorder = TraceRecorder({})
    miss = recorder.emit(0, "a", "BatchExistKey", keys=["key"])
    start = recorder.emit(1, "a", "BatchPutStart", keys=["key"], value_sizes=[1])
    end = recorder.emit(2, "a", "BatchPutEnd", keys=["key"], put_start=start)
    read1 = recorder.emit(3, "b", "BatchGetReplicaList", keys=["key"])
    read2 = recorder.emit(3, "c", "BatchGetReplicaList", keys=["key"])
    recorder.emit(4, "a", "BatchRemove", keys=["key"])
    assert recorder.events[1]["depends_on"] == [miss]
    assert recorder.events[2]["depends_on"] == [start]
    assert recorder.events[3]["depends_on"] == [end]
    assert recorder.events[4]["depends_on"] == [end]
    assert set(recorder.events[5]["depends_on"]) == {end, read1, read2}


def test_lifecycle_capacity_is_independent_of_request_clients(tmp_path):
    recorder = TraceRecorder({"seed": 42})
    recorder.setup([f"worker-{i}" for i in range(8)], 2, 1 << 30)
    recorder.emit(100, "worker-0", "BatchExistKey", keys=["missing"])
    path = tmp_path / "rpc.jsonl"
    recorder.write(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["metadata"]["storage"]["total_bytes"] == 2 << 30
    mounts = [r for r in rows[1:] if r["op"] == "MountSegment"]
    unmounts = [r for r in rows[1:] if r["op"] == "UnmountSegment"]
    assert len(mounts) == len(unmounts) == 2
    assert {r["segment_id"] for r in mounts} == {r["segment_id"] for r in unmounts}
    assert all(r["phase"] == "setup" and r["depends_on"] for r in mounts)
    assert all(r["phase"] == "teardown" for r in unmounts)
    assert all(r["segments"] == [] for r in rows[1:] if r["op"] == "ReMountSegment")
