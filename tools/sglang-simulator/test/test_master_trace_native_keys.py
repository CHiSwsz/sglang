from types import SimpleNamespace

from sglang_simulator.master_trace.storage import MHALayout


def test_physical_keys_match_native_mooncake_backend():
    from sglang_simulator.simulation.sglang.hook_bootstrap import (
        install_simulator_hooks,
    )

    install_simulator_hooks()
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

    layout = MHALayout("Qwen/Qwen3-8B", 36, 8, 128, 2, 256, 8)
    for rank in range(8):
        native = object.__new__(MooncakeStore)
        native.config_prefix = "Qwen-Qwen3-8B"
        native.is_mla_backend = False
        native.should_split_heads = False
        native.mha_suffix = str(rank)
        observed = []
        native.store = SimpleNamespace(
            batch_is_exist=lambda keys: observed.extend(keys) or [1] * len(keys)
        )
        hashes = ["chain-hash-1", "chain-hash-2"]
        assert native.batch_exists(hashes) == 2
        assert observed == layout.keys(hashes, rank)


def test_hybrid_restoration_intersects_checkpoint_boundaries_across_ranks():
    from sglang_simulator.simulation.sglang.hook_bootstrap import (
        install_simulator_hooks,
    )

    install_simulator_hooks()
    from sglang_simulator.master_trace.backend import RecordingBackend
    from sglang_simulator.master_trace.storage import Entry, SharedStorage
    from sglang_simulator.master_trace.trace import TraceRecorder

    from sglang.srt.mem_cache.hicache_storage import (
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
    )

    layout = MHALayout("Qwen/Qwen3.8-27B", 16, 4, 256, 2, 256, 8)
    storage = SharedStorage(TraceRecorder({}), layout, 1 << 30, 1 << 30)
    backend = RecordingBackend(storage, 0)
    backend.register_mem_host_pool_v2(SimpleNamespace(kv_buffer=object()), PoolName.KV)
    backend.register_mem_host_pool_v2(
        SimpleNamespace(conv_buffer=[object()], temporal_state_elem_size=1),
        PoolName.MAMBA,
    )
    hashes = ["page1", "page2", "page3"]
    for rank in range(8):
        for key in layout.keys(hashes, rank):
            storage.entries[key] = Entry(1, pending=False)
        checkpoints = ["page1", "page3"] if rank == 0 else ["page2"]
        for page in checkpoints:
            for component in ("temporal", "conv_0"):
                key = f"Qwen-Qwen3.8-27B_{page}_{rank}_{component}"
                storage.entries[key] = Entry(1, pending=False)
    transfer = PoolTransfer(
        name=PoolName.MAMBA, hit_policy=PoolHitPolicy.TRAILING_PAGES
    )
    result = backend.batch_exists_v2(hashes, [transfer])
    assert (
        result.kv_hit_pages == 0
    )  # Taking min(per-rank maximum) would wrongly return 2.
    assert result.restorable_prefix_pages == []
    for rank in range(1, 8):
        for component in ("temporal", "conv_0"):
            key = f"Qwen-Qwen3.8-27B_page1_{rank}_{component}"
            storage.entries[key] = Entry(1, pending=False)
    result = backend.batch_exists_v2(hashes, [transfer])
    assert result.kv_hit_pages == 1
    assert result.restorable_prefix_pages == [1]
