"""Connect native HiCache decisions to the shared, virtual-time storage model."""

from queue import Empty

from greenlet import getcurrent, greenlet

from .storage import SharedStorage

_construction_context = None
_simulator_kernel_library = None


def install_simulator_import_stubs():
    """Allow CPU-only model discovery; model math must never execute here."""
    import torch

    global _simulator_kernel_library
    if _simulator_kernel_library is None:
        library = torch.library.Library("sgl_kernel", "FRAGMENT")
        schemas = [
            "fused_sigmoid_mul_cpu(Tensor(a!) input, Tensor gate) -> ()",
            "fused_qk_gemma_rmsnorm_cpu(Tensor q, Tensor k, Tensor q_weight, Tensor k_weight, float eps, int head_dim) -> (Tensor, Tensor)",
            "fused_qk_gemma_rmsnorm_with_gate_cpu(Tensor q_gate, Tensor k, Tensor q_weight, Tensor k_weight, float eps, int head_dim, int num_head) -> (Tensor, Tensor, Tensor)",
            "fused_qkvzba_split_reshape_cat_contiguous_cpu(Tensor mixed_qkvz, Tensor mixed_ba, int num_heads_qk, int num_heads_v, int head_qk, int head_v) -> (Tensor, Tensor, Tensor, Tensor)",
        ]

        def unavailable(*args):
            raise RuntimeError(
                "Model math cannot run in the metadata-only trace driver"
            )

        for schema in schemas:
            name = schema.split("(", 1)[0]
            if not hasattr(torch.ops.sgl_kernel, name):
                library.define(schema)
                library.impl(name, unavailable, "CPU")
        _simulator_kernel_library = library

    # Retain native extra-buffer policy checks while bypassing only its real
    # accelerator requirement. No GPU kernels execute in this driver.
    from unittest.mock import patch

    from sglang.srt.arg_groups import mamba_hook

    original = mamba_hook.validate_mamba_extra_buffer
    if not getattr(original, "_master_trace_platform", False):

        def validate(*args, **kwargs):
            platform = mamba_hook.get_platform()

            class ModeledAccelerator:
                is_cuda = True

                def __getattr__(self, name):
                    return getattr(platform, name)

            with patch.object(mamba_hook, "get_platform", lambda: ModeledAccelerator()):
                return original(*args, **kwargs)

        validate._master_trace_platform = True
        mamba_hook.validate_mamba_extra_buffer = validate


def install_metadata_only_device_payload():
    """Keep native allocation indices on CPU without allocating unused KV data."""
    import torch

    # Native decode prepares pinned staging tensors even with CPU execution.
    # Preserve their values and indices, without requesting a CUDA allocator.
    tensor = torch.tensor
    if not getattr(tensor, "_master_trace_unpinned", False):

        def unpinned_tensor(*args, **kwargs):
            if kwargs.get("pin_memory"):
                kwargs["pin_memory"] = False
            return tensor(*args, **kwargs)

        unpinned_tensor._master_trace_unpinned = True
        torch.tensor = unpinned_tensor
        torch.Tensor.pin_memory = lambda self, *args, **kwargs: self

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    original = MHATokenToKVPool._create_buffers_normal
    if getattr(original, "_master_trace_meta_payload", False):
        return

    def allocate(self):
        device = self.device
        try:
            self.device = "meta"
            return original(self)
        finally:
            self.device = device

    allocate._master_trace_meta_payload = True
    MHATokenToKVPool._create_buffers_normal = allocate

    from unittest.mock import patch

    import torch

    from sglang.srt.mem_cache.memory_pool import MambaPool

    original_mamba = MambaPool.__init__
    if not getattr(original_mamba, "_master_trace_meta_payload", False):

        def allocate_mamba(self, *args, **kwargs):
            zeros = torch.zeros

            def state_zeros(*dims, **options):
                shape = options.get("size", dims[0] if len(dims) == 1 else dims)
                if isinstance(shape, (tuple, list)) and len(shape) >= 3:
                    options["device"] = "meta"
                return zeros(*dims, **options)

            with patch.object(torch, "zeros", state_zeros):
                return original_mamba(self, *args, **kwargs)

        allocate_mamba._master_trace_meta_payload = True
        MambaPool.__init__ = allocate_mamba


def bind_instance(storage: SharedStorage, instance: int):
    global _construction_context
    _construction_context = (storage, instance)


def create_backend():
    if _construction_context is None:
        raise RuntimeError("recording backend requires a cluster driver")
    return RecordingBackend(*_construction_context)


class RecordingBackend:
    def __init__(self, storage, instance):
        self.storage = storage
        self.instance = instance
        self.last_completion_us = 0
        self.registered_pools = {}
        self.active_flow = None
        self.flows = {}

    def run_flow(self, name, action):
        flow = self.flows.get(name)
        if flow is None or flow.task.dead:
            flow = StorageFlow(self, name, action)
            self.flows[name] = flow
            flow.resume()

    def wait_until(self, done):
        if done > self.storage.now_us:
            if self.active_flow is None:
                raise RuntimeError("blocking storage call requires an execution flow")
            getcurrent().parent.switch(done)

    def register_mem_pool_host(self, pool):
        self.pool = pool

    def register_mem_host_pool_v2(self, pool, name):
        if str(name) not in ("kv", "mamba"):
            raise ValueError(f"unsupported trace pool: {name}")
        self.registered_pools[name] = pool
        if str(name) == "kv":
            self.pool = pool

    def _native(self, rank):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        native = object.__new__(MooncakeStore)
        native.config_prefix = self.storage.layout.model_name.replace("/", "-")
        native.is_mla_backend = False
        native.should_split_heads = False
        native.mha_suffix = str(rank)
        native.mem_pool_host = self.pool
        native.registered_pools = self.registered_pools
        client = self.storage.layout.clients(self.instance)[rank]
        native._batch_exist = lambda keys, extra_info=None: self.storage.lookup_keys(
            client, keys
        )
        return native

    def batch_exists(self, keys, extra_info=None):
        return self.storage.lookup(self.instance, keys)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        from sglang.srt.mem_cache.hicache_storage import PoolTransferResult

        if pool_transfers:
            results = [
                self._native(rank).batch_exists_v2(keys, pool_transfers, extra_info)
                for rank in range(self.storage.layout.tp_size)
            ]
            restorable = set(results[0].restorable_prefix_pages or [])
            for result in results[1:]:
                restorable.intersection_update(result.restorable_prefix_pages or [])
            pages = max(restorable, default=0)
            names = set().union(*(result.extra_pool_hit_pages for result in results))
            return PoolTransferResult(
                pages,
                {
                    name: min(
                        result.extra_pool_hit_pages.get(name, 0) for result in results
                    )
                    for name in names
                },
                sorted(restorable),
            )
        return PoolTransferResult(
            kv_hit_pages=self.batch_exists(keys), extra_pool_hit_pages={}
        )

    def batch_set(self, keys, values=None, extra_info=None):
        self.last_completion_us = self.storage.write(self.instance, keys)
        self.wait_until(self.last_completion_us)
        return True

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self.batch_set(keys)
        return [True] * len(keys)

    def batch_set_v2(self, transfers, extra_info=None):
        return self._sidecar_io(transfers, True)

    def batch_get_v2(self, transfers, extra_info=None):
        return self._sidecar_io(transfers, False)

    def _sidecar_io(self, transfers, write):
        results = {}
        for transfer in transfers:
            completion = self.storage.now_us
            sizes = self.storage.sidecar_component_sizes[str(transfer.name)]
            page_hits = [True] * len(transfer.keys)
            for rank, client in enumerate(self.storage.layout.clients(self.instance)):
                native = self._native(rank)
                keys, multiplier = native._get_hybrid_page_component_keys(
                    transfer.keys, transfer
                )
                keys = native._tag_keys(keys)
                if multiplier != len(sizes):
                    raise RuntimeError(
                        "sidecar component count does not match native Mooncake keys"
                    )
                if write:
                    exists = self.storage.lookup_keys(client, keys)
                    missing = [i for i, state in enumerate(exists) if state != 1]
                    if missing:
                        done = self.storage.write_keys(
                            client,
                            [keys[i] for i in missing],
                            [sizes[i % multiplier] for i in missing],
                        )
                        completion = max(completion, done)
                else:
                    done, found = self.storage.read_keys(client, keys)
                    for i in range(len(page_hits)):
                        page_hits[i] &= all(
                            found[i * multiplier : (i + 1) * multiplier]
                        )
                    completion = max(completion, done)
            self.last_completion_us = completion
            # A blocking native transfer must finish before the next pool or
            # batch in this flow is submitted. Logical TP ranks join here.
            self.wait_until(completion)
            results[transfer.name] = page_hits
        return results

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        done, hits = self.storage.read(self.instance, keys, return_hits=True)
        self.last_completion_us = max(self.last_completion_us, done)
        self.wait_until(done)
        return hits

    def close(self):
        pass


class StorageFlow:
    """Suspend a native synchronous call stack on the shared logical clock."""

    def __init__(self, backend, name, action):
        self.backend = backend
        self.name = name
        self.task = greenlet(action)

    def resume(self):
        backend = self.backend
        previous = backend.active_flow
        backend.active_flow = self
        try:
            with backend.storage.recorder.stream(self.name):
                done = self.task.switch()
        finally:
            backend.active_flow = previous
        if not self.task.dead:
            if done <= backend.storage.now_us:
                raise RuntimeError("storage flow must resume in the future")
            backend.storage._schedule(done, self.resume)


def _drain(queue):
    if queue is None:
        return
    while True:
        try:
            operation = queue.get_nowait()
        except Empty:
            return
        if operation is not None:
            yield operation


def install_recording_controller(target):
    # Disable ALL storage worker threads, including the subclass's overridden
    # backup thread. Otherwise those threads race the discrete-event driver.
    for name in (
        "prefetch_thread_func",
        "prefetch_io_aux_func",
        "prefetch_sync_thread_func",
        "backup_thread_func",
    ):
        setattr(target, name, lambda self: None)

    def backup(self):
        if not self.enable_storage:
            return
        backend = self.storage_backend

        def consume():
            for operation in _drain(self.backup_queue):
                self._page_backup(operation)
                # Native synchronous calls have now finished in simulated time.
                self.ack_backup_queue.put(operation)

        backend.run_flow("backup", consume)

    def prefetch(self):
        if not self.enable_storage:
            return
        from unittest.mock import patch

        from sglang.srt.managers.cache_controller import PrefetchAck

        backend = self.storage_backend

        class Acknowledgements:
            def put(queue, ack):
                self._reduce_prefetch_ack(ack)
                self.ack_prefetch_queue.put(ack)

        def query():
            for operation in _drain(self.prefetch_queue):
                if operation.is_terminated():
                    self.prefetch_revoke_queue.put(operation.request_id)
                    continue
                hashes, hits = self._storage_hit_query(operation)
                operation.hash_value = hashes[: hits // self.page_size]
                operation.storage_hit_count = hits
                self.prefetch_hit_queue.put(operation)

        def read():
            for operation in _drain(self.prefetch_buffer):
                if operation.is_terminated():
                    self.ack_prefetch_queue.put(
                        PrefetchAck(
                            rid=operation.request_id,
                            operation=operation,
                            completed_req=True,
                        )
                    )
                    continue
                with (
                    patch.object(self, "prefetch_sync_queue", Acknowledgements()),
                    patch.object(self, "page_get_func", self._page_get_zero_copy),
                ):
                    self._page_transfer(operation)
                    self.prefetch_sync_queue.put(
                        PrefetchAck(
                            rid=operation.request_id,
                            operation=operation,
                            completed_req=True,
                        )
                    )

        backend.run_flow("prefetch_query", query)
        backend.run_flow("prefetch_read", read)

    target.handle_backup_operation = backup
    target.handle_prefetch_operation = prefetch
