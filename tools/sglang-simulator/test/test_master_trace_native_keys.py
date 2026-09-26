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
