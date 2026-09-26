# Offline Mooncake master RPC producer

This experimental adapter drives one native SGLang CPU-simulator scheduler per
serving instance, with separate L1/L2 caches and a shared logical-time L3 model.
It exports RPC intent for a separate Mooncake replay tool. No Mooncake master,
GPU, model weights or payload transfer is used during generation.

The supported layout is homogeneous MHA, BF16, page-first, PP=1, CP=1, with KV
heads divisible by TP. A singleton real CPU/Gloo group executes native scheduler
and HiCache decisions; each simulated instance's storage operations expand into
the physical K/V keys of its TP ranks. TP ranks are not independently simulated.
MLA, hybrid sidecar pools, group semantics, heterogeneous layouts and failover
are unsupported. The adapter lives in this SGLang fork; Mooncake requires only
its versioned output contract, not a dependency on this fork.

## Generate

Use the official CPU simulator's environment and source checkout. Add
`tools/sglang-simulator/src`, `python` and the checkout root to `PYTHONPATH`.
The checked environment used Python 3.12, CPU PyTorch 2.14, transformers 5.12 and
AIC 0.10. SGLang's import dependencies are still needed even with a replay timing
predictor. The model path can point to `test/assets/qwen3-8b` with dummy weights.

```bash
python -m sglang_simulator.master_trace.workload \
  --config /outside/repo/workload.json --output /outside/repo/requests.jsonl
python -m sglang_simulator.master_trace.cluster \
  --config /outside/repo/cluster.json --requests /outside/repo/requests.jsonl \
  --output-dir /outside/repo/recording
```

The output directory must not exist. Workload configuration, for example:

```json
{"seed":42,"sessions":12,"session_rate":8,"length_variation":0,"mean_new_tokens_per_round":[1024,256,256],"mean_return_tokens_per_round":[8,8,8],"mean_inter_round_interval_ms":[0,2000,2000],"round_ratios":[0,0,1]}
```

This borrows bench_mix's weighted session lengths and accumulated conversation
history. It exports token IDs in AutoBench JSONL: `prompt`, `prompt_len`,
`output_len`, `timestamp` in milliseconds and session/turn metadata. New sessions
arrive as a Poisson process. Later turns append the preceding simulated response
(token 1) and new random input. Turn timestamps are open-loop offsets; generation
fails if a later turn would arrive before the previous response completes.
No semantic equivalence between different token strings is assumed.

Cluster configuration, for example:

```json
{"model_path":"/checkout/tools/sglang-simulator/test/assets/qwen3-8b","model_name":"Qwen/Qwen3-8B","sim_config_path":"simulator.json","instances":2,"tp_size":8,"page_size":256,"max_total_tokens":4096,"max_running_requests":8,"context_length":8192,"hicache_ratio":2,"storage_nodes":2,"storage_bytes_per_node":68719476736,"storage_bandwidth_bytes_per_s":17179869184,"storage_latency_us":100}
```

Model and simulator configuration paths resolve relative to the cluster JSON.
`sim_config_path` uses the official simulator configuration schema, including
platform, predictor and modeled TP size. Calibrate the predictor before claiming
GPU-equivalent throughput. A synthetic replay latency table only validates the
pipeline and defines an assumed offered load.

Routing is `(session_id + turn) % instances`, deliberately moving successive
turns between caches. It is a reproducible reuse scenario, not an implementation
of a production load balancer. HiCache uses write-through and wait-complete
prefetch; native prefix hashing and storage batch boundaries are retained.

## Timing, state and lifecycle

One discrete-event clock orders all instances and storage completions. Scheduler
CPU overhead is explicitly zero instead of the generator's wall-clock runtime.
The real GPU is replaced by the official simulator's modeled batch duration.
L2 load timing uses the simulator; L2 backup is acknowledged by its CPU path and
is not separately delayed by this adapter. GPU/CPU overlap and multi-rank skew
are approximate. L3 uses a full-duplex aggregate FIFO bandwidth model, not a
packet-level network model. These limitations must accompany reported results.

Pending L3 writes become visible only at transfer completion. Reads pin resident
objects; capacity eviction removes unpinned committed objects in LRU order.
This is an explicit workload policy, not a reproduction of Mooncake's placement
or eviction algorithms. Model sufficient capacity for correctness runs and
inspect actual allocation/status counters in replay. Exhaustion with no eligible
victim fails generation instead of silently inventing a successful write.

Each MHA page creates K and V objects per TP rank. Each object's bytes are
`layers * (kv_heads / TP) * head_dim * dtype_bytes * page_size`. Physical key
format is compared against the native MooncakeStore backend in a unit test.
Per-key write/read dependencies preserve visibility and anti-dependencies;
concurrent read calls remain independent. PutEnd retains its matching Start.

Exported v2 traces explicitly register all request and storage clients using
empty ReMountSegment handshakes, mount configured storage segments in setup,
run workload, then unmount all segments after a completion barrier. Each phase
has its own microsecond origin. Serving client count does not determine storage
capacity. Segment addresses and IDs are supplied by the real replayer, and no
payload memory is allocated there. Dynamic membership is outside this version.

Outputs are `master-rpc.jsonl` and `simulation.json`, including configuration,
input SHA-256, completed requests, logical duration, and L3 counts. Keep these
artifacts outside the Mooncake repository. Archive source revisions and the
timing configuration alongside them, then stop this process before benchmarking.
