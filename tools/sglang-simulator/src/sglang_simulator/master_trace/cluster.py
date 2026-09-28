"""Drive native SGLang schedulers on one shared discrete-event clock.

Experimental homogeneous MHA/TP deployments, with one native CPU scheduler per
logical serving instance. The official simulator replaces GPU execution; the
recorder expands each instance's storage operations into physical TP-rank keys.
"""

import argparse
import hashlib
import heapq
import json
import math
import os
import time
from array import array
from dataclasses import asdict
from pathlib import Path

from .arrivals import SessionTimeline
from .storage import MHALayout, SharedStorage
from .trace import TraceRecorder


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Cluster:
    def __init__(self, config, rows, output_dir):
        os.environ["SGLANG_SIMULATOR_EXTERNAL_CLOCK"] = "1"
        os.environ["SGLANG_SIMULATOR_CPU_OVERHEAD_US"] = "0"
        os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"
        os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(output_dir)
        os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = config["sim_config_path"]

        from sglang_simulator.simulation.sglang.hook_bootstrap import (
            install_simulator_hooks,
        )

        install_simulator_hooks()
        import torch

        from .backend import install_simulator_import_stubs

        install_simulator_import_stubs()
        from sglang_simulator.compat import SIMULATOR_SERVER_ARG_OVERRIDES
        from sglang_simulator.simulation.manager import StateManager
        from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

        from sglang.srt.distributed import bootstrap
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.runtime_context import SpawnRanks, publish
        from sglang.srt.server_args import PortArgs, ServerArgs

        from .backend import bind_instance, install_metadata_only_device_payload

        torch.set_num_threads(1)
        install_metadata_only_device_payload()
        self.torch = torch
        self.state_manager = StateManager
        self.scheduler_hook = C_SchedulerHook
        self.config = config
        self.timeline = SessionTimeline(rows)
        self.rows = list(self.timeline.requests.values())
        self.now = 0
        self.queue = []
        self.sequence = 0
        self.completed = {}
        self.request_rows = self.timeline.requests
        self.iterations = 0
        model = json.loads((Path(config["model_path"]) / "config.json").read_text())
        model = model.get("text_config", model)
        if model.get("dtype", model.get("torch_dtype")) != "bfloat16":
            raise ValueError("the trace driver currently requires a BF16 model")
        self.layout = MHALayout(
            config["model_name"],
            model.get("layer_types", []).count("full_attention")
            if "layer_types" in model
            else model["num_hidden_layers"],
            model["num_key_value_heads"],
            model.get("head_dim", model["hidden_size"] // model["num_attention_heads"]),
            2,
            config["page_size"],
            config["tp_size"],
        )
        self.recorder = TraceRecorder(
            {
                "producer": "sglang-native-cpu-cluster",
                "config": config,
                "layout": asdict(self.layout),
                "logical_gpus": config["instances"] * config["tp_size"],
                "clock": "unified_discrete_event",
                "storage_call_order": "serial per (client_id, stream_id); blocking storage calls wait for simulated transfer completion",
                "storage_streams": ["backup", "prefetch_query", "prefetch_read"],
                "metadata_rpc_latency_us": 0,
                "link_model": "full_duplex_aggregate_FIFO",
                "initial_state": "empty",
                "device_kv_payload": "meta_tensor; native CPU allocation indices retained",
                "storage_eviction_mode": config.get(
                    "storage_eviction_mode", "record_remove"
                ),
                "storage_state_semantics": "simulated LRU; replay master may choose different victims and read outcomes",
            }
        )
        clients = [
            client
            for i in range(config["instances"])
            for client in self.layout.clients(i)
        ]
        self.recorder.setup(
            clients, config["storage_nodes"], config["storage_bytes_per_node"]
        )
        self.storage = SharedStorage(
            self.recorder,
            self.layout,
            config["storage_nodes"] * config["storage_bytes_per_node"],
            config["storage_bandwidth_bytes_per_s"],
            config.get("storage_latency_us", 100),
            eviction_mode=config.get("storage_eviction_mode", "record_remove"),
        )
        self.storage.sidecar_component_sizes = {}
        linear_layers = model.get("layer_types", []).count("linear_attention")
        if linear_layers:
            from types import SimpleNamespace

            from sglang.srt.configs.mamba_utils import (
                Mamba2StateShape,
                mamba2_state_dtype,
            )

            if model.get("model_type") != "qwen3_5_text":
                raise ValueError("only Qwen3.5/3.8 GDN hybrid state is supported")
            state_dtype = mamba2_state_dtype(SimpleNamespace(**model))

            shape = Mamba2StateShape.create(
                tp_world_size=config["tp_size"],
                intermediate_size=model["linear_value_head_dim"]
                * model["linear_num_value_heads"],
                n_groups=model["linear_num_key_heads"],
                num_heads=model["linear_num_value_heads"],
                head_dim=model["linear_value_head_dim"],
                state_size=model["linear_key_head_dim"],
                conv_kernel=model["linear_conv_kernel_dim"],
            )
            self.storage.sidecar_component_sizes["mamba"] = [
                linear_layers
                * math.prod(shape.temporal)
                * state_dtype.temporal.itemsize,
                *[
                    linear_layers * math.prod(conv) * state_dtype.conv.itemsize
                    for conv in shape.conv
                ],
            ]
            self.recorder.metadata["sidecar_component_sizes"] = (
                self.storage.sidecar_component_sizes
            )
            self.recorder.metadata["sidecar_dtypes"] = {
                "temporal": str(state_dtype.temporal),
                "conv": str(state_dtype.conv),
            }
        kwargs = dict(
            model_path=config["model_path"],
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_ratio=config["hicache_ratio"],
            # The factory is replaced by RecordingBackend. Select file here to
            # avoid initializing a real Mooncake transfer engine at bootstrap.
            hicache_write_policy="write_through",
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
            max_total_tokens=config["max_total_tokens"],
            page_size=config["page_size"],
            max_running_requests=config.get("max_running_requests", 8),
            context_length=config.get("context_length", 8192),
            skip_tokenizer_init=True,
            log_level="error",
            watchdog_timeout=36000,
            **SIMULATOR_SERVER_ARG_OVERRIDES,
        )
        if linear_layers:
            kwargs.update(
                max_mamba_cache_size=config.get("max_mamba_cache_size", 1024),
            )
        server_args = ServerArgs(**kwargs)
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(world_rank=0, dp_rank=0, gpu_id=0),
        )
        original_bootstrap = bootstrap.init_parallel_runtime

        def shared_cpu_group(**kwargs):
            if not bootstrap._PARALLEL_INITIALISED:
                original_bootstrap(**kwargs)

        # Every simulated instance uses an independent cache/scheduler, but
        # real CPU execution is singleton TP=1 and shares its Gloo process group.
        bootstrap.init_parallel_runtime = shared_cpu_group
        self.instances = []
        for index in range(config["instances"]):
            StateManager.reset()
            bind_instance(self.storage, index)
            scheduler = Scheduler(server_args, PortArgs.init_new(server_args))
            type(scheduler.output_streamer).stream_output = (
                lambda streamer, *args, **kwargs: self.on_output(*args, **kwargs)
            )
            self.instances.append(
                {
                    "scheduler": scheduler,
                    "state": self.capture_state(),
                    "inbox": [],
                    "busy": False,
                    "wake": None,
                }
            )
            print(f"Initialized instance {index + 1}/{config['instances']}", flush=True)
        self.initial_state = self.capture_state()
        for rid, row in self.timeline.initial:
            when = round(row["timestamp"] * 1000)
            self.enqueue(when, "arrival", (rid, row))

    def capture_state(self):
        return {
            key: value
            for key, value in vars(self.state_manager).items()
            if key.startswith("_") and isinstance(value, (int, float))
        }

    def activate(self, instance):
        for key, value in instance["state"].items():
            setattr(self.state_manager, key, value)
        self.state_manager.set_global_clock(self.now / 1_000_000)

    def enqueue(self, when, kind, payload):
        self.sequence += 1
        heapq.heappush(self.queue, (when, self.sequence, kind, payload))

    def wake(self, index, when=None):
        instance = self.instances[index]
        when = self.now if when is None else when
        if instance["busy"] or (
            instance["wake"] is not None and instance["wake"] <= when
        ):
            return
        instance["wake"] = when
        self.enqueue(when, "step", index)

    def on_output(self, reqs, *args, **kwargs):
        for req in reqs:
            if req.finished() and req.rid not in self.completed:
                self.completed[req.rid] = {
                    "finished_us": self.now,
                    "output_ids": list(req.output_ids),
                }
                following = self.timeline.complete(req.rid, self.now)
                if following is not None:
                    self.enqueue(
                        round(following[1]["timestamp"] * 1000), "arrival", following
                    )
                if len(self.completed) % 1024 == 0:
                    print(
                        f"Completed {len(self.completed)}/{len(self.rows)} requests at {self.now / 1e6:.3f}s",
                        flush=True,
                    )

    def admit(self, scheduler, rid, row):
        from sglang_simulator.simulation.sglang.req_stats_manager import (
            request_stats_manager,
        )

        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
        from sglang.srt.sampling.sampling_params import SamplingParams

        params = SamplingParams(
            max_new_tokens=row["output_len"], ignore_eos=True, temperature=0
        )
        params.normalize(None)
        request = TokenizedGenerateReqInput(
            rid=rid,
            input_text=None,
            input_ids=array("q", row["prompt"]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=params,
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
        )
        stats = request_stats_manager.get_req_stats(rid)
        stats.created_time = row["timestamp"] / 1000
        stats.queue_start = self.now / 1_000_000
        stats.last_event_time = stats.created_time
        stats.input_length = row["prompt_len"]
        stats.output_length = row["output_len"]
        scheduler.handle_generate_request(request)

    def step(self, index):
        instance = self.instances[index]
        if instance["wake"] != self.now or instance["busy"]:
            return
        instance["wake"] = None
        self.activate(instance)
        scheduler = instance["scheduler"]
        for rid, row in instance["inbox"]:
            self.admit(scheduler, rid, row)
        instance["inbox"].clear()
        plan = scheduler.get_next_batch_to_run(
            scheduler.running_batch, scheduler.last_batch
        )
        scheduler.running_batch = plan.running_batch
        batch = plan.batch_to_run
        if batch is not None:
            scheduler.cur_batch_for_debug = batch
            result = scheduler.run_batch(batch)
            latency = (
                self.state_manager.get_current_inference_dur()
                + self.state_manager._hicache_l2_load_dur
            )
            if not math.isfinite(latency) or latency <= 0:
                raise RuntimeError(f"invalid predicted batch latency: {latency}")
            instance["busy"] = True
            instance["batch"] = (batch, result, self.scheduler_hook.SIMULATION_BATCH)
            self.enqueue(self.now + math.ceil(latency * 1_000_000), "complete", index)
            self.iterations += 1
        else:
            scheduler.last_batch = None
            if scheduler.waiting_queue or scheduler.running_batch.reqs:
                # Completion callbacks wake the owning instance exactly when
                # its I/O finishes. Do not poll on other instances' rank I/O.
                self.wake(index, self.now + 1000)
        instance["state"] = self.capture_state()

    def finish(self, index):
        instance = self.instances[index]
        self.activate(instance)
        scheduler = instance["scheduler"]
        batch, result, simulation_batch = instance.pop("batch")
        self.scheduler_hook.SIMULATION_BATCH = simulation_batch
        scheduler.process_batch_result(batch, result)
        scheduler.last_batch = batch
        instance["busy"] = False
        instance["state"] = self.capture_state()
        self.wake(index)

    def run(self):
        with self.torch.inference_mode():
            while self.queue or self.storage.jobs:
                next_event = self.queue[0][0] if self.queue else math.inf
                next_io = self.storage.next_completion()
                self.now = min(next_event, next_io if next_io is not None else math.inf)
                self.storage.advance(self.now)
                for index, instance in enumerate(self.instances):
                    cc = instance["scheduler"].tree_cache.cache_controller
                    if (
                        not cc.ack_backup_queue.empty()
                        or not cc.ack_prefetch_queue.empty()
                    ):
                        self.wake(index)
                while self.queue and self.queue[0][0] == self.now:
                    _, _, kind, payload = heapq.heappop(self.queue)
                    if kind == "arrival":
                        rid, row = payload
                        meta = row.get("metadata", {})
                        index = (meta.get("session_id", 0) + meta.get("turn", 0)) % len(
                            self.instances
                        )
                        self.instances[index]["inbox"].append((rid, row))
                        self.wake(index)
                    elif kind == "step":
                        self.step(payload)
                    else:
                        self.finish(payload)
                if self.now > self.config.get("max_simulation_us", 120_000_000):
                    raise RuntimeError(
                        "simulation exceeded configured horizon; inspect stalled requests"
                    )
        if len(self.completed) != len(self.rows):
            raise RuntimeError(
                f"only {len(self.completed)}/{len(self.rows)} requests completed"
            )
        for rid, completed in self.completed.items():
            expected = self.request_rows[rid]["output_len"]
            if completed["output_ids"] != [1] * expected:
                raise RuntimeError(f"unexpected simulated response for {rid}")
        self.timeline.validate()

    def report(self):
        from sglang_simulator.simulation.sglang.req_stats_manager import (
            request_stats_manager,
        )

        return {
            "completed_requests": len(self.completed),
            "iterations": self.iterations,
            "simulation_duration_us": self.now,
            "storage": self.storage.stats,
            "requests": [
                {
                    "id": rid,
                    "arrival_us": round(self.request_rows[rid]["timestamp"] * 1000),
                    "session": self.request_rows[rid].get("metadata", {}),
                    **completed,
                    **asdict(request_stats_manager.get_req_stats(rid)),
                }
                for rid, completed in self.completed.items()
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    for field in ("model_path", "sim_config_path"):
        config[field] = str((args.config.parent / config[field]).resolve())
    with args.requests.open() as stream:
        rows = []
        for line in stream:
            if line.strip():
                row = json.loads(line)
                row["prompt"] = array("I", row["prompt"])
                rows.append(row)
    start = time.monotonic()
    cluster = Cluster(config, rows, args.output_dir)
    cluster.recorder.metadata["input_sha256"] = file_sha256(args.requests)
    cluster.run()
    resolved_path = args.output_dir / "requests.autobench.jsonl"
    with resolved_path.open("x") as stream:
        for row in cluster.timeline.resolved_rows():
            stream.write(json.dumps(row, separators=(",", ":"), default=list) + "\n")
    cluster.recorder.metadata["resolved_requests_sha256"] = file_sha256(resolved_path)
    cluster.recorder.metadata["storage_stats"] = cluster.storage.stats
    cluster.recorder.write(args.output_dir / "master-rpc.jsonl")
    report = cluster.report()
    report["generation_wall_seconds"] = time.monotonic() - start
    (args.output_dir / "simulation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "requests"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
