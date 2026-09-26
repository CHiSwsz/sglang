"""Shared metadata-only L3 with deterministic asynchronous transfers.

The bandwidth model is a FIFO aggregate link per direction (full duplex),
not a packet-level network simulator. Pending writes are invisible to reads;
pending writes and pinned reads cannot be evicted. This model produces RPC
intent, while the benchmark measures a separate, real Mooncake master.
"""

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass

from .trace import TraceRecorder


@dataclass(frozen=True)
class MHALayout:
    model_name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int
    page_size: int
    tp_size: int

    def __post_init__(self):
        if (
            min(
                self.layers,
                self.kv_heads,
                self.head_dim,
                self.dtype_bytes,
                self.page_size,
                self.tp_size,
            )
            < 1
        ):
            raise ValueError("KV layout dimensions must be positive")
        if self.kv_heads % self.tp_size:
            raise ValueError("initial recorder requires KV heads divisible by TP")

    @property
    def object_bytes(self):
        # MHA page_first: one object for K and one for V, each containing all
        # layers for this TP rank. Matches MooncakeStore._get_mha_buffer_meta.
        return (
            self.layers
            * (self.kv_heads // self.tp_size)
            * self.head_dim
            * self.dtype_bytes
            * self.page_size
        )

    def keys(self, hashes, rank):
        prefix = self.model_name.replace("/", "-")
        return [f"{prefix}_{key}_{rank}_{kv}" for key in hashes for kv in ("k", "v")]

    def clients(self, instance):
        return [
            f"instance-{instance:04d}/tp-{rank:02d}" for rank in range(self.tp_size)
        ]


@dataclass
class Entry:
    size: int
    pending: bool = True
    pins: int = 0


class SharedStorage:
    def __init__(
        self,
        recorder: TraceRecorder,
        layout: MHALayout,
        capacity_bytes,
        bandwidth_bytes_per_s,
        latency_us=100,
    ):
        if capacity_bytes < 1 or bandwidth_bytes_per_s <= 0 or latency_us < 0:
            raise ValueError("invalid storage capacity, bandwidth or latency")
        self.recorder = recorder
        self.layout = layout
        self.capacity = capacity_bytes
        self.bandwidth = bandwidth_bytes_per_s
        self.latency_us = latency_us
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.now_us = 0
        self.jobs = []
        self.sequence = 0
        self.link_ready = {"read": 0, "write": 0}
        self.stats = {
            "lookup_keys": 0,
            "hit_keys": 0,
            "pending_keys": 0,
            "evicted_keys": 0,
            "written_keys": 0,
            "read_keys": 0,
            "peak_bytes": 0,
        }

    def advance(self, now_us):
        if now_us < self.now_us:
            raise ValueError("shared storage clock cannot go backwards")
        while self.jobs and self.jobs[0][0] <= now_us:
            when, _, callback = heapq.heappop(self.jobs)
            self.now_us = when
            callback()
        self.now_us = now_us

    def next_completion(self):
        return self.jobs[0][0] if self.jobs else None

    def _schedule(self, when, callback):
        self.sequence += 1
        heapq.heappush(self.jobs, (when, self.sequence, callback))

    def _transfer_end(self, direction, size):
        start = max(self.now_us + self.latency_us, self.link_ready[direction])
        finish = start + math.ceil(size * 1_000_000 / self.bandwidth)
        self.link_ready[direction] = finish
        return finish

    def lookup(self, instance, hashes):
        hit_pages = len(hashes)
        for rank, client in enumerate(self.layout.clients(instance)):
            keys = self.layout.keys(hashes, rank)
            self.recorder.emit(self.now_us, client, "BatchExistKey", keys=keys)
            rank_hits = 0
            prefix_open = True
            for index, key in enumerate(keys):
                entry = self.entries.get(key)
                self.stats["lookup_keys"] += 1
                if entry is not None and not entry.pending:
                    self.stats["hit_keys"] += 1
                    self.entries.move_to_end(key)
                    if prefix_open:
                        rank_hits = index + 1
                else:
                    prefix_open = False
                    if entry is not None:
                        self.stats["pending_keys"] += 1
            hit_pages = min(hit_pages, rank_hits // 2)
        return hit_pages

    def _reserve(self, client, new_keys, protected):
        needed = len(new_keys) * self.layout.object_bytes
        victims = []
        freed = 0
        for key, entry in self.entries.items():
            if self.used_bytes + needed - freed <= self.capacity:
                break
            if key not in protected and not entry.pending and not entry.pins:
                victims.append(key)
                freed += entry.size
        if self.used_bytes + needed - freed > self.capacity:
            raise RuntimeError("L3 has insufficient unpinned capacity for this backup")
        dependencies = []
        if victims:
            dependencies.append(
                self.recorder.emit(self.now_us, client, "BatchRemove", keys=victims)
            )
            for key in victims:
                self.used_bytes -= self.entries.pop(key).size
            self.stats["evicted_keys"] += len(victims)
        for key in new_keys:
            self.entries[key] = Entry(self.layout.object_bytes)
            self.used_bytes += self.layout.object_bytes
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.used_bytes)
        return dependencies

    def write(self, instance, hashes):
        completions = []
        protected = {
            key
            for rank in range(self.layout.tp_size)
            for key in self.layout.keys(hashes, rank)
        }
        for rank, client in enumerate(self.layout.clients(instance)):
            keys = self.layout.keys(hashes, rank)
            new_keys = [key for key in keys if key not in self.entries]
            dependencies = self._reserve(client, new_keys, protected)
            start = self.recorder.emit(
                self.now_us,
                client,
                "BatchPutStart",
                keys=keys,
                value_sizes=[self.layout.object_bytes] * len(keys),
                depends_on=dependencies,
            )
            done = self._transfer_end("write", len(new_keys) * self.layout.object_bytes)

            def finish(client=client, keys=keys, new_keys=new_keys, start=start):
                for key in new_keys:
                    self.entries[key].pending = False
                self.stats["written_keys"] += len(new_keys)
                self.recorder.emit(
                    self.now_us, client, "BatchPutEnd", keys=keys, put_start=start
                )

            self._schedule(done, finish)
            completions.append(done)
        return max(completions, default=self.now_us)

    def read(self, instance, hashes):
        completions = []
        for rank, client in enumerate(self.layout.clients(instance)):
            keys = self.layout.keys(hashes, rank)
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or entry.pending:
                    raise RuntimeError(
                        "prefetch target disappeared or is not committed"
                    )
                entry.pins += 1
                self.entries.move_to_end(key)
            self.recorder.emit(self.now_us, client, "BatchGetReplicaList", keys=keys)
            done = self._transfer_end("read", len(keys) * self.layout.object_bytes)

            def finish(keys=keys):
                for key in keys:
                    self.entries[key].pins -= 1
                self.stats["read_keys"] += len(keys)

            self._schedule(done, finish)
            completions.append(done)
        return max(completions, default=self.now_us)
