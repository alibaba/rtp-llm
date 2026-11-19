# RTP-LLM Unified Performance Benchmark

Use `tools/perf_test` for local, SSH and AOne model-level prefill/decode tests.
Each run takes one strict JSON config. Declare the model, topology, engine,
workload and output in that file. Mode is inferred from `workload.kind`; action
is `run` for a config and `report` for an existing run directory. Explicit
`--mode` remains an assertion and must agree with the workload.

## Configure and preview

```bash
./tools/perf_test --config=rtp_llm/test/perf_test/examples/prefill_local.json --dry-run
./tools/perf_test --config=rtp_llm/test/perf_test/examples/decode_ssh_gang.json --dry-run
./tools/perf_test --config=rtp_llm/test/perf_test/examples/prefill_aone_ci.json --dry-run
```

Remove `--dry-run` to execute. Preview validates the complete plan without
creating a run directory, contacting hosts, building or starting a model.

| Backend | Configuration | Execution |
| --- | --- | --- |
| local | `execution.backend="local"`, `devices` | Current machine |
| ssh | `execution.backend="ssh"`, `hosts`, `gpus_per_machine`, `source` | Named remote machines |
| aone | `execution.backend="aone"`, resource and GPU type | AOne shards or gang |

`model.topology` is the only user declaration of TP/DP/PP/EP. Workload kind
`cache_grid` selects prefill; `grid` and `distribution` select decode. Decode
uses fixed global batches divisible by DP. Current model-level measurements
require PP=1. Prefill cache grids and decode profiling require DP=1.
For ordinary decode, `workload.profiling.steps` must be between 1 and
`min(1000, shortest output length)` in a single capture round; increasing
`runs` does not extend this limit. SP/MTP decode currently requires `steps=1`:
output tokens do not establish a reliable model-independent iteration budget.
These checks apply to both `engine.args.sp_type` and its paired `engine.env.SP_TYPE`;
an explicit CLI argument takes precedence over the environment value.
Unsafe windows are rejected during planning, before starting an engine.
Each capture round re-arms the participating ranks with `start_profile`, then
profiling waits for every round's trace on each local participating rank.
Local traces use the run's `engine_traces/` directory, independent of ambient
profiler environment settings.

## Shared local and AOne configuration

`examples/prefill_shared.json` uses schema version 2. Model, engine and workload
are shared. `execution.local` and `execution.aone` hold backend-specific settings;
only the selected backend participates in the resolved execution plan. Version-1
local/AOne configs remain supported. SSH keeps its version-1 contract and runner.

```bash
# Same file, local or AOne single-machine execution.
./tools/perf_test --config=rtp_llm/test/perf_test/examples/prefill_shared.json --dry-run
./tools/perf_test --config=rtp_llm/test/perf_test/examples/prefill_shared.json \
  --set execution.backend=aone --dry-run

# The example has TP=2, DP=1: one GPU on each of two AOne hosts.
./tools/perf_test --config=rtp_llm/test/perf_test/examples/prefill_shared.json \
  --set execution.backend=aone \
  --set execution.aone.gpus_per_machine=1 \
  --set execution.aone.shard_count=null --dry-run
```

Remove `--dry-run` to execute, after setting model/tokenizer paths accessible on
the selected machines. Switching backends does not change model topology or
workload. Existing capability restrictions still apply, including gang-only
AOne decode. AOne case sharding and model gang execution are mutually exclusive.

`--set PATH=VALUE` applies typed overrides without changing the input file:
JSON numbers, booleans, arrays, objects and null are parsed as JSON; other values
are strings. Unknown structured fields are errors. Use a whole object for map
keys containing dots, for example
`--set 'execution.aone.exec_properties={"custom.pool":"benchmark"}'`.
Scheduler properties must have string values. Generated `gpu` and `gpu_count`
cannot conflict with explicit properties; gangs retain full-host reservations.
For experiments, override the base with paths such as
`--set base_config.execution.backend=local`.

## Bazel platform environment

The CLI submits local measurements to the internal
`//rtp_llm/test/perf_test:perf_test_local` py_test. AOne single-machine and gang
measurements already run as Bazel tests. `execution.platform` selects the real
Bazel `--config`, so `.bazelrc` test environments supply CC, CXX, CUDAHOSTCXX,
PATH and LD_LIBRARY_PATH, and the existing NVIDIA wrapper adds runfiles library
paths and obtains GPU locks. No new environment variables are required.

Version-2 local configs require a platform; AOne can infer it from GPU type and
rejects incompatible explicit platforms. Legacy local configs default to cuda13.
The selected toolchain and model dependencies must exist on execution machines.
`engine.env` remains available for benchmark-specific settings and explicit
overrides; it is not necessary to duplicate platform toolchain settings there.

Use `--bazel-arg=--jobs=8` and
`--bazel-startup-arg=--output_user_root=/absolute/worktree-cache` for native
Bazel options. Local submissions use a worktree/platform-specific output root by
default. Platform, worker input, GPU allocation and test timeout are managed
by perf_test; conflicting raw options are rejected. The internal local target
is manual/local and cannot submit AOne jobs. The redundant `:perf_test` binary
has been removed; use `tools/perf_test`.

`execution.test_timeout_seconds` drives both Bazel timeout and the existing
worker watchdog (default 7000 seconds). Explicit existing budgets, including
19500 seconds in the DeepSeek daily examples, are retained. Local device IDs
index the wrapper's locked allocation; the launcher requests enough slots for
the configured IDs and overrides inherited WORLD_SIZE for allocation.

CLI result roots are resolved from the submitting working directory before
entering runfiles. Local results stay under that persistent root, with submission
bundles and command evidence under its `.launch/` subdirectory. Workers consume
frozen plans without re-reading configs or re-sampling workloads. Runs retain
`input_config.json`, `launch_request.json` and, for local workers,
`launch_environment.json` with selected actual toolchain environment values.
Later resume/profile invocations keep the original run evidence and write their
own request/environment records alongside the corresponding submission bundle.
Resume/profile use persisted plans; offline report and dry-run do not launch a
GPU test. `--set` is only accepted for new config runs.

## Prefill batches

`examples/prefill_multi_grid_local.json` contains named cold, explicit mixed,
random and traffic grids in one engine session. Freeze its sampled batches
before sharing a config:

```bash
python -m rtp_llm.test.perf_test.prefill.export \
  --config rtp_llm/test/perf_test/examples/prefill_multi_grid_local.json \
  --output /tmp/prefill-batches-frozen.json
```

The exporter detects random and traffic sources, including named grids.
`--output-dir DIR` instead exports one file per batch size for a single
top-level random workload. Existing destinations are rejected. The corresponding
Bazel binary is `//rtp_llm/test/perf_test:export_prefill_batches`.

## SSH machines

Start with `examples/decode_ssh_gang.json` under `rtp_llm/test/perf_test/` for
SSH execution settings. For prefill, combine its `execution` object with the
model, engine and workload from `examples/prefill_local.json` in one JSON.
Set topology, `gpus_per_machine` and host count together; the local prefill
example needs one host. There is no launcher YAML or referenced benchmark JSON
to maintain.

Each host has a `ssh_target` for the submitting machine's SSH connection and
an `advertise_host` reachable by the other model workers. System SSH config,
including aliases and jump hosts, is respected. Passwordless access and a
prepared Python/GPU environment are required. Credentials do not belong in
benchmark configs.

World size is TP × DP × PP. The per-machine width is the smaller of world size
and `gpus_per_machine`; world size must divide evenly across hosts. Supply
exactly the required number of hosts, in rank order. A single remote host is
supported. Model paths must be available in the remote runtime.

`source.ref="auto"` freezes the submitting public worktree's current Git commit.
Commit tracked changes first. A branch/tag must already resolve locally; it
is frozen to a full SHA before submission. Only committed Git objects are sent
to executor-owned remote mirrors/checkouts. Untracked local files are not sent.

Build policies are `build_with_deps`, `build` and `reuse`. The first two require
an installed `source.build_driver` implementing the build request/receipt
protocol. Machine admission prepares that adapter and its execution policy;
the backend does not silently run bare Bazel when it is missing. Reuse requires
matching code, build settings, toolchain and verified native artifacts.

The worker owns only its run's process group and allocated devices. Cancellation
and cleanup use the recorded owner identity, rather than process-name matching.
A disconnected submitter can inspect or cancel its run later:

```bash
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=status
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=cancel
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=collect
./tools/perf_test --run-dir=/path/to/ssh-run --maintenance=clean
```

Collection retains failure evidence and does not promote an interrupted run to
success. Cleanup refuses active or mismatched owners and preserves source/build
caches. SSH supports prefill run and collected offline report, and decode run;
SSH checkpoint resume and standalone profile are not enabled.

## Batch experiments

`examples/decode_ssh_experiment.json` contains a complete inline `base_config`
and named benchmarks. `overrides` changes explicit fields, `zip_axes` combines
equal-length arrays by index, and `product_axes` expands a Cartesian product.
Overlapping paths and backend/source/output overrides are rejected.

```bash
./tools/perf_test --config=rtp_llm/test/perf_test/examples/decode_ssh_experiment.json --dry-run
```

Every expanded task is planned before the first remote side effect. Input files
are snapshotted once; each task's final workload and seed are materialized once.
Execution uses those frozen plans in order. Each task has its own run and result;
`experiment_plan.json` and `experiment_summary.json` index the experiment.

## Results and follow-up actions

Runs are written under `output.result_root/<run_id>/`. Read
`results/result.json` for the schema-v2 result envelope, `results/summary.json`
for aggregate metrics and `reports/` for reports. Prefill records cases in
`results/cases.jsonl`; decode uses `results/cases/`. Profiles and traces retain
member provenance. Execution plans use the current schema only; old plan
versions and removed launcher entries are rejected.

Decode `throughput_tokens_per_second` counts successful output tokens divided
by the client wall-clock duration of each concurrent round. It includes HTTP
transport, request dispatch, and distributed coordination; it is not a pure
engine compute rate. Worker threads and HTTP sessions are prepared before
formal timing and reused across rounds. Warmup rounds also warm the connections;
with `warmup_runs=0`, the first measured round includes initial TCP setup.
Latency fields continue to come from the server's `aux_info`.

```bash
./tools/perf_test --run-dir=/path/to/local-prefill-run --action=resume
./tools/perf_test --run-dir=/path/to/local-prefill-run --action=profile --cases=CASE_ID
./tools/perf_test --run-dir=/path/to/prefill-run --action=report
```

Local prefill supports run/resume/profile/report. AOne and SSH prefill support
run and collected offline report. Decode supports run, including its automatic
report; AOne decode requires a multi-machine gang. Follow-up actions read the
frozen run and do not accept config overrides.

The manual `offline_bench_test` is a separate microbenchmark. Public runs do not
automatically publish results or replace regression baselines. Enabling a
`publish` backend is rejected until an execution path supports publication;
credentials and signed URLs must remain outside benchmark configs.

## Trace analysis

Prefill and decode store profile manifests in
`profiles/<session_id>/manifest.json` and captured files in
`traces/<session_id>/<case_id>/`. Schema 2 manifests list each case's capture
status and run-relative artifact path, format, rank and phase (unknown values
are null). Case dimensions and topology come from `execution_plan.json`.

Analyze a captured run, an experiment, or a directory containing runs:

```bash
python -m rtp_llm.test.perf_test.multi_node.analyzer.batch_analyze /path/to/run
```

The command writes `reports/trace/analysis.json` and `kernels.csv` below its
input directory; `--output-dir DIR` selects another destination. It does not
launch an engine or change measurements. Missing, incomplete, invalid and
unsupported artifacts are listed in `skipped`. An input with no analyzable
trace exits nonzero. Copied collected runs are supported without rewriting
the original plan.

Only listed Kineto/Chrome Trace JSON files are analyzed. `.nsys-rep` captures
remain available for Nsight Systems and are reported as unsupported by this
JSON analyzer. Each trace retains its session/case/rank context. Kernel time is
the sum of raw kernel durations; kernel span is the first-to-last kernel
interval. Neither is end-to-end request latency. No duration outliers are
silently removed. `--min-occurrences N` filters table rows only (default 1).

For a single trace, use
`python -m rtp_llm.test.perf_test.multi_node.analyzer.trace_analyze /path/to/trace.json`.
Optional module analysis uses the restored configs with the same run input:

```bash
python -m rtp_llm.test.perf_test.multi_node.analyzer.batch_analyze /path/to/run \
  --module-conf rtp_llm/test/perf_test/multi_node/analyzer/configs/qwen_480b_prefill_fp8.py
```

`module_conf.py` provides name matching only; the two `qwen_480b_*` presets also
provide model-specific FLOPs/IO estimates. Select a preset matching the model
and phase; the supplied formulas describe TP1. `modules.csv` and each trace's
`statistics.modules` contain matched kernel names, actual invocation counts,
summed durations and percentages of total kernel time. Rules match independently,
so overlapping patterns can contribute to more than one module. Kernel row
filtering does not remove events from module totals.

FLOPs/IO describe one module invocation. Rates are computed only when the spec
explicitly declares `formula_per_kernel=True`: estimated work per invocation
is multiplied by the actual matching kernel-event count and divided by summed
duration. QKV/O GEMM presets declare this; other presets leave rates null when
logical module invocations cannot be inferred from kernel launches. Context comes from the frozen case and
topology, with a per-DP batch and the case's input length. Heterogeneous or cached
prefill shapes, unsupported TP/phase, invalid formulas and unknown capture steps
produce null estimates and an `estimation_error`, while measured module times
remain available. Decode rates require explicit decode phase metadata. These
estimates are not measured hardware counters or end-to-end latency. Config files
are Python files defining `module_config` with strings or `ModuleConf` values.
The old Task-directory scanner and shell wrappers remain removed.

## Result goldens

Generate a candidate from a completed result, review the differences, then
validate the accepted file explicitly:

```bash
python -m rtp_llm.test.perf_test.regression.golden generate \
  --result /path/to/run/results/result.json --output /tmp/result.candidate.json
python -m rtp_llm.test.perf_test.regression.golden validate \
  --result /path/to/run/results/result.json --golden /path/to/accepted.golden.json
```

`generate` atomically replaces its output; use a candidate path before promotion.
`validate` reads the golden without changing it. The corresponding Bazel binary
is `//rtp_llm/test/perf_test:perf_test_golden`.
