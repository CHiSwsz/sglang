"""Resolve session plans against simulated response completion, without GPU code."""

import math


class SessionTimeline:
    def __init__(self, rows):
        self.requests = {
            f"request-{index:08d}": {**row, "metadata": dict(row.get("metadata", {}))}
            for index, row in enumerate(rows)
        }
        self.initial = []
        self.followers = {}
        self.finished = {}
        sessions = {}
        for rid, row in self.requests.items():
            meta = row["metadata"]
            if "session_id" in meta:
                turn = meta.get("turn", 0)
                if not isinstance(turn, int) or turn < 0:
                    raise ValueError("session turns must be nonnegative integers")
                turns = sessions.setdefault(meta["session_id"], {})
                if turn in turns:
                    raise ValueError("duplicate session turn")
                turns[turn] = rid
            mode = meta.get("arrival_mode", "absolute")
            if mode == "after_completion":
                delay = meta.get("think_time_us")
                if (
                    "session_id" not in meta
                    or meta.get("turn", 0) == 0
                    or not isinstance(delay, int)
                    or delay < 0
                    or row.get("timestamp") is not None
                ):
                    raise ValueError(
                        "relative turns require a session, delay and unresolved timestamp"
                    )
            elif mode == "absolute":
                timestamp = row.get("timestamp")
                if (
                    not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp)
                    or timestamp < 0
                ):
                    raise ValueError(
                        "absolute arrivals require a finite nonnegative timestamp"
                    )
                self.initial.append((rid, row))
            else:
                raise ValueError(f"unsupported arrival mode: {mode}")
        self.predecessors = {}
        for turns in sessions.values():
            if sorted(turns) != list(range(len(turns))):
                raise ValueError("session turns must be contiguous starting at zero")
            for turn, rid in turns.items():
                if turn:
                    previous = turns[turn - 1]
                    self.predecessors[rid] = previous
                    if (
                        self.requests[rid]["metadata"].get("arrival_mode")
                        == "after_completion"
                    ):
                        self.followers[previous] = rid

    def complete(self, rid, finished_us):
        if rid in self.finished:
            raise ValueError(f"duplicate completion: {rid}")
        timestamp = self.requests[rid]["timestamp"]
        if timestamp is None or finished_us < round(timestamp * 1000):
            raise ValueError("completion precedes request arrival")
        self.finished[rid] = finished_us
        following = self.followers.get(rid)
        if following is None:
            return None
        row = self.requests[following]
        when = finished_us + row["metadata"]["think_time_us"]
        row["timestamp"] = when / 1000
        return following, row

    def validate(self):
        if len(self.finished) != len(self.requests):
            raise RuntimeError("not all session turns completed")
        for rid, previous in self.predecessors.items():
            row = self.requests[rid]
            arrival_us = round(row["timestamp"] * 1000)
            previous_end = self.finished[previous]
            if row["metadata"].get("arrival_mode") == "after_completion":
                if arrival_us != previous_end + row["metadata"]["think_time_us"]:
                    raise RuntimeError(
                        "resolved arrival does not match completion plus think time"
                    )
            elif arrival_us < previous_end:
                raise RuntimeError(
                    "session arrival precedes previous completion; use a completion-relative plan"
                )

    def resolved_rows(self):
        self.validate()
        resolved = []
        for rid, row in self.requests.items():
            meta = {**row["metadata"], "trace_request_id": rid}
            if meta.get("arrival_mode") == "after_completion":
                meta["resolved_from"] = meta.pop("arrival_mode")
            resolved.append({**row, "metadata": meta})
        return sorted(
            resolved,
            key=lambda row: (row["timestamp"], row["metadata"]["trace_request_id"]),
        )
