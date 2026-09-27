# Offline Mooncake master RPC producer

This experimental adapter drives one native SGLang CPU-simulator scheduler per
serving instance, with separate L1/L2 caches and a shared logical-time L3 model.
It exports RPC intent for a separate Mooncake replay tool. No Mooncake master,
GPU, model weights or payload transfer is used during generation.

The supported layout is homogeneous MHA/GQA, BF16, page-first, PP=1, CP=1, with
KV heads and TP dividing one another (including replicated KV heads when TP is
larger). Each K/V object must fit one Mooncake slice (at most
4 MiB minus 16 bytes for the pinned CacheLib build); larger objects are rejected.
A singleton real CPU/Gloo group executes native scheduler
and HiCache decisions; each simulated instance's storage operations expand into
the physical K/V keys of its TP ranks. TP ranks are not independently simulated.
Qwen3.5/3.8 GDN hybrid models also record their Mamba temporal and convolution
checkpoint objects. Other hybrid layouts, MLA, group semantics, heterogeneous
layouts and failover are unsupported. The adapter lives in this SGLang fork; Mooncake requires only
its versioned output contract, not a dependency on this fork.

## Generate

Use the official CPU simulator's environment and source checkout. Add
`tools/sglang-simulator/src`, `python` and the checkout root to `PYTHONPATH`.
The checked environment used Python 3.12, CPU PyTorch 2.14, transformers 5.12 and
AIC 0.10. SGLang's import dependencies are still needed even with a replay timing
predictor. The model path can point to `test/assets/qwen3-8b` with dummy weights.

```bash
python -m sglang_simulator.master_trace.workload \
  --config /outside/repo/workload.json --output /outside/repo/session-plan.jsonl
python -m sglang_simulator.master_trace.cluster \
  --config /outside/repo/cluster.json --requests /outside/repo/session-plan.jsonl \
  --output-dir /outside/repo/recording
```

The output directory must not exist. Workload configuration, for example:

```json
{"seed":42,"session_arrival_duration_s":10,"session_rate":2,"length_variation":0.2,"mean_new_tokens_per_round":[1024,256,256],"mean_return_tokens_per_round":[8,12,16],"mean_think_time_ms":[0,1000,2000],"think_time_cv":1.0,"round_ratios":[2,3,5]}
```

This borrows bench_mix's weighted session lengths, accumulated history and
completion-relative followups, not its exact arrival generator or distributions.
New sessions arrive as a Poisson process throughout `session_arrival_duration_s`;
alternatively set a positive `sessions` count, but not both. The simulation drains
all existing sessions after new arrivals stop. Each session independently samples
its total number of rounds using `round_ratios`. Input and output lengths vary
uniformly around their per-round means by `length_variation`.

For every later turn, think time is sampled from a lognormal distribution with
the configured arithmetic `mean_think_time_ms` and coefficient of variation
`think_time_cv` (standard deviation / mean). Index zero is unused. A zero mean
allows immediate followups; CV zero gives a fixed wait. These are explicit
synthetic assumptions, not parameters fitted to production traffic. Random
new-session arrivals use a separate stream from lengths and think times.

The session plan contains token IDs, `prompt_len`, `output_len` and session/turn
metadata. Only first turns have absolute millisecond `timestamp` values. Later
turns have `timestamp: null`, `arrival_mode: after_completion` and a sampled
`think_time_us` in metadata. This plan is **not yet an AutoBench arrival trace**.
The cluster schedules each followup at the previous simulated response completion
plus its think time. Slower responses therefore delay that session's next turn,
while independent new-session arrivals continue. Later prompts append the previous
simulated response (token 1) and new random input; response tokens are validated.
No semantic equivalence between different token strings is assumed.

After simulation, `requests.autobench.jsonl` contains every resolved absolute
timestamp, sorted by arrival, in AutoBench format. It can be loaded as fixed
arrivals without adding think time again. The input plan and resolved trace have
separate SHA-256 hashes in the RPC header. Existing absolute AutoBench inputs
remain supported. For reproducing old phase-separated experiments only, the
legacy `mean_inter_round_interval_ms` option retains fixed arrival offsets and
rejects overlap with an unfinished previous turn; do not combine it with think time.

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
An object missing at read time returns a failed read through the native HiCache
acknowledgement and fallback path, even if an earlier availability query hit.
This is an explicit workload policy, not a reproduction of Mooncake's placement
or eviction algorithms. Model sufficient capacity for correctness runs and
inspect actual allocation/status counters in replay. Exhaustion with no eligible
victim fails generation instead of silently inventing a successful write.

Set `storage_eviction_mode` to `master` for autonomous master-eviction pressure
runs. The simulator still applies its LRU capacity policy to generate subsequent
request intents, but does not export capacity victims as client `BatchRemove`
calls. Mounted capacity is unchanged. The real master's background thread must
select and remove victims; measure its eviction counters and allocation failures.
Recorded reads may miss and repeated writes may already exist because its
victims differ from the simulated LRU. This remains fixed-intent replay, without
feedback from the measured master into the simulator. The default
`record_remove` preserves explicit simulated removal calls.

Device KV and Mamba payload buffers are meta tensors, as are the simulated host payload
buffers. Native CPU request/page indices and cache metadata remain allocated.
This avoids retaining one unused placeholder KV payload for every simulated
instance; it does not reduce logical cache capacity or recorded value sizes.
Pinned staging tensors use ordinary CPU memory. Accelerator-only validation is
bypassed inside the native extra-buffer validator; its cache policy checks remain.
Missing CPU model-import operators have fail-fast stubs: executing model math is
an error. These adaptations are confined to this experimental trace driver.

Each MHA page creates K and V objects per TP rank. Each object's bytes are
`full_attention_layers * max(1, kv_heads / TP) * head_dim * dtype_bytes * page_size`. Physical key
format is compared against the native MooncakeStore backend in a unit test.
Per-key write/read dependencies preserve visibility and anti-dependencies;
concurrent read calls remain independent. PutEnd retains its matching Start.

For the checked Qwen3.8-27B configuration, 16 full-attention layers, 4 KV heads,
head dimension 256, TP=8 and page size 256 give 2 MiB per K or V object. All
eight ranks store their physical objects, including replicated heads. The 48
linear-attention layers contribute an 18 MiB temporal object (FP32) and a
368,640-byte convolution object (BF16) per rank at each native offload checkpoint.
Checkpoint placement, trailing-state hit queries and read/write batch boundaries
come from native HiCache and MooncakeStore. A KV prefix is reusable only at a
checkpoint boundary restorable on every rank; taking the minimum rank maximum
would be incorrect when checkpoint sets have holes.

State shapes and dtypes come from SGLang's Mamba configuration helpers. Objects
larger than the pinned CacheLib client's 4 MiB minus 16 byte slice limit use the
replay contract's `value_slices`, preserving one logical object across slices.
The native page transfer path emits KV and sidecar completion acknowledgements
on the shared virtual clock. The default `max_mamba_cache_size` is 1024; set it
in cluster configuration to model a different state-slot budget. Long prompts
are held as compact token arrays; output JSON and token values are unchanged.

Exported v2 traces explicitly register all request and storage clients using
empty ReMountSegment handshakes, mount configured storage segments in setup,
run workload, then unmount all segments after a completion barrier. Each phase
has its own microsecond origin. Serving client count does not determine storage
capacity. Segment addresses and IDs are supplied by the real replayer, and no
payload memory is allocated there. Dynamic membership is outside this version.

Heartbeats are explicit Ping events generated offline: one after each client
registration, then every 1,000,000 microseconds per client during workload, with
evenly staggered client offsets. This synthetic schedule covers both serving
and storage clients and ends at the last workload timestamp. Its policy is
recorded in header metadata. The replayer uses the original Ping timestamps;
the shared worker pool can delay them. There is no runtime heartbeat generator
or extension while overdue RPCs drain, so overload may cause real client expiry.

Outputs are `master-rpc.jsonl`, `requests.autobench.jsonl` and `simulation.json`,
including configuration, input/resolved SHA-256, actual arrivals, session metadata,
completed requests, logical duration, and L3 counts. Keep these
artifacts outside the Mooncake repository. Archive source revisions and the
timing configuration alongside them, then stop this process before benchmarking.
