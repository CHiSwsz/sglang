"""Version 2 RPC-intent export with explicit, independent storage lifecycle.

Phases are replayed in order with completion barriers. Each phase has its own
relative microsecond clock. Mount time is consequently measured independently
of serving traffic, and teardown cannot race unfinished writes or lookups.
"""

import json
from contextlib import contextmanager
from pathlib import Path

HEARTBEAT_INTERVAL_US = 1_000_000


def with_heartbeats(events, interval_us=HEARTBEAT_INTERVAL_US):
    """Add deterministic Ping intents offline, without changing business events.

    Ping each client after registration, then stagger periodic workload Pings
    evenly across one interval. Stop at the last recorded workload timestamp;
    replay delays never extend this finite schedule.
    """
    if not isinstance(interval_us, int) or interval_us <= 0:
        raise ValueError("heartbeat interval must be a positive integer")
    clients = []
    previous_ping = {}
    next_ping = 0
    for event in events:
        if event["op"] == "Ping":
            raise ValueError("input already contains Ping events")
        if event["phase"] == "workload":
            if not clients:
                raise ValueError("workload requires registered clients")
            while next_ping * interval_us // len(clients) <= event["timestamp_us"]:
                client = clients[next_ping % len(clients)]
                ping = {
                    "id": f"ping-workload-{next_ping}",
                    "phase": "workload",
                    "timestamp_us": next_ping * interval_us // len(clients),
                    "client_id": client,
                    "op": "Ping",
                    "stream_id": "heartbeat",
                    "depends_on": [previous_ping[client]],
                }
                yield ping
                previous_ping[client] = ping["id"]
                next_ping += 1
        yield event
        if event["op"] == "ReMountSegment":
            client = event["client_id"]
            clients.append(client)
            previous_ping[client] = f"ping-setup-{len(clients) - 1}"
            yield {
                "id": f"ping-setup-{len(clients) - 1}",
                "phase": "setup",
                "timestamp_us": event["timestamp_us"],
                "client_id": client,
                "op": "Ping",
                "stream_id": "heartbeat",
                "depends_on": [event["id"]],
            }


class TraceRecorder:
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.events = []
        self._counter = 0
        self._storage = []
        self._writers = {}
        self._readers = {}
        self.current_stream = None
        self._stream_tail = {}

    @contextmanager
    def stream(self, name):
        previous = self.current_stream
        self.current_stream = name
        try:
            yield
        finally:
            self.current_stream = previous

    def emit(self, timestamp_us, client_id, op, *, phase="workload", **fields):
        if timestamp_us < 0:
            raise ValueError("negative event time")
        event_id = f"e{self._counter}"
        self._counter += 1
        dependencies = set(fields.get("depends_on", ()))
        if phase == "workload":
            name = self.current_stream or "default"
            fields["stream_id"] = name
            stream = (client_id, name)
            if stream in self._stream_tail:
                dependencies.add(self._stream_tail[stream])
            self._stream_tail[stream] = event_id
        keys = fields.get("keys", ())
        if keys:
            read = op in ("BatchExistKey", "BatchGetReplicaList")
            for key in keys:
                if key in self._writers:
                    dependencies.add(self._writers[key])
                if read:
                    self._readers.setdefault(key, set()).add(event_id)
                else:
                    dependencies.update(self._readers.pop(key, ()))
                    self._writers[key] = event_id
        if dependencies:
            fields["depends_on"] = sorted(
                dependencies, key=lambda value: int(value[1:])
            )
        self.events.append(
            {
                "id": event_id,
                "phase": phase,
                "timestamp_us": round(timestamp_us),
                "client_id": client_id,
                "op": op,
                **fields,
            }
        )
        return event_id

    def setup(self, request_clients, storage_nodes, bytes_per_node):
        if storage_nodes < 1 or bytes_per_node < 1 or self.events:
            raise ValueError("setup requires positive capacity and an empty recorder")
        storage_clients = [f"storage-{i:04d}" for i in range(storage_nodes)]
        registrations = {}
        for client in list(request_clients) + storage_clients:
            # Empty remount is the master's initial client-liveness handshake;
            # request-only clients contribute no storage capacity.
            registrations[client] = self.emit(
                0, client, "ReMountSegment", phase="setup", segments=[]
            )
        for index, client in enumerate(storage_clients):
            segment_id = f"segment-{index:04d}"
            self.emit(
                0,
                client,
                "MountSegment",
                phase="setup",
                segment_id=segment_id,
                size_bytes=bytes_per_node,
                depends_on=[registrations[client]],
            )
            self._storage.append((client, segment_id))
        self.metadata["storage"] = {
            "nodes": storage_nodes,
            "bytes_per_node": bytes_per_node,
            "total_bytes": storage_nodes * bytes_per_node,
            "lifecycle": "mount before workload; drain workload; unmount all segments",
            "payload": "metadata_only",
        }

    def write(self, path: Path):
        if not self._storage:
            raise ValueError("storage setup is required")
        events = list(self.events)
        for index, (client, segment_id) in enumerate(self._storage):
            events.append(
                {
                    "id": f"unmount-{index}",
                    "phase": "teardown",
                    "timestamp_us": 0,
                    "client_id": client,
                    "op": "UnmountSegment",
                    "segment_id": segment_id,
                }
            )
        phase_order = {"setup": 0, "workload": 1, "teardown": 2}
        events.sort(key=lambda e: (phase_order[e["phase"]], e["timestamp_us"]))
        self.metadata["heartbeats"] = {
            "source": "offline_synthetic",
            "interval_us": HEARTBEAT_INTERVAL_US,
            "setup": "one Ping after each registration",
            "workload": "periodic per client with evenly staggered offsets",
            "end": "last workload timestamp; no extension during replay",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "master_rpc_trace",
                        "version": 2,
                        "time_unit": "us",
                        "metadata": self.metadata,
                    }
                )
                + "\n"
            )
            for event in with_heartbeats(events):
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
