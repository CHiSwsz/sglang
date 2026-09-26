"""Connect native HiCache decisions to the shared, virtual-time storage model."""

from queue import Empty

from .storage import SharedStorage

_construction_context = None


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

    def register_mem_pool_host(self, pool):
        self.pool = pool

    def register_mem_host_pool_v2(self, pool, name):
        if str(name) != "kv":
            raise ValueError("initial trace adapter supports the MHA KV pool only")
        self.pool = pool

    def batch_exists(self, keys, extra_info=None):
        return self.storage.lookup(self.instance, keys)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        from sglang.srt.mem_cache.hicache_storage import PoolTransferResult

        if pool_transfers:
            raise ValueError("sidecar pools are not supported by the MHA recorder")
        return PoolTransferResult(
            kv_hit_pages=self.batch_exists(keys), extra_pool_hit_pages={}
        )

    def batch_set(self, keys, values=None, extra_info=None):
        self.last_completion_us = self.storage.write(self.instance, keys)
        return True

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self.batch_set(keys)
        return [True] * len(keys)

    def batch_set_v2(self, transfers, extra_info=None):
        if transfers:
            raise ValueError("sidecar pools are not supported by the MHA recorder")
        return {}

    def close(self):
        pass


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
        for operation in _drain(self.backup_queue):
            backend.last_completion_us = backend.storage.now_us
            self._page_backup(operation)  # Native hashing, batching and policy.
            completed = operation.completed_tokens
            operation.completed_tokens = 0

            def acknowledge(operation=operation, completed=completed):
                operation.completed_tokens = completed
                self.ack_backup_queue.put(operation)

            backend.storage._schedule(backend.last_completion_us, acknowledge)

    def prefetch(self):
        if not self.enable_storage:
            return
        from sglang.srt.managers.cache_controller import STORAGE_BATCH_SIZE, PrefetchAck

        backend = self.storage_backend
        for operation in _drain(self.prefetch_queue):
            if operation.is_terminated():
                self.prefetch_revoke_queue.put(operation.request_id)
                continue
            hashes, hits = self._storage_hit_query(operation)
            operation.hash_value = hashes[: hits // self.page_size]
            operation.storage_hit_count = hits
            self.prefetch_hit_queue.put(operation)

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
            done = backend.storage.now_us
            for offset in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
                hashes = operation.hash_value[offset : offset + STORAGE_BATCH_SIZE]
                done = backend.storage.read(backend.instance, hashes)

            def acknowledge(operation=operation):
                self.ack_prefetch_queue.put(
                    PrefetchAck(
                        rid=operation.request_id,
                        operation=operation,
                        completed_tokens=len(operation.hash_value) * self.page_size,
                        completed_req=True,
                    )
                )

            backend.storage._schedule(done, acknowledge)

    target.handle_backup_operation = backup
    target.handle_prefetch_operation = prefetch
