# RTP-LLM Unified Performance Benchmark

RTP-LLM uses one strict performance-test entry for local prefill and decode:
`tools/perf_test`. New runs accept one versioned JSON config; engine arguments,
environment, topology, workload and output are declared in that file rather
than forwarded as command-line flags. Regression and publishing sections remain
accepted for schema compatibility; public runs do not invoke those helpers.

## Migrating from the legacy perf CLI

The removed `perf_test` test target and `test_entry.py` flags do not work with
`tools/perf_test`. Start from a JSON file in `rtp_llm/test/perf_test/examples/`
and map the old options as follows:

| Legacy option | Unified equivalent |
| --- | --- |
| `--ckpt_path`, `--tokenizer_path`, `--model_type` | `model.checkpoint`, `model.tokenizer`, `model.type` in the JSON config |
| `--dp_size`, `--tp_size`, `--pp_size`, `--ep_size` | `model.topology.dp`, `.tp`, `.pp`, `.ep` |
| `--batch_size`, `--input_len`, `--output_len` | Decode `workload.batch.sizes`, `workload.input_lengths`, `workload.output_lengths`; prefill uses `workload.batch_sizes` and its cache-grid axes |
| `--test_env=NAME=value` | Supported environment settings in `engine.env`; undeclared settings are rejected |
| `--perf_test_name` | `model.label` for the descriptive name; the run ID is generated separately |
| `--partial=1` (decode), `--partial=2` (prefill) | `--mode=decode` or `--mode=prefill`; `--partial=0` requires two runs |

The unified entry does not download ModelScope or Hugging Face repository IDs.
Download the model and tokenizer before running it, then set `model.checkpoint`
and `model.tokenizer` to paths available in the runtime. The separate manual
`offline_bench_test` still resolves Hub IDs; that behavior is not part of the
unified entry.

## Single-Node Benchmark

Preview cache-aware prefill without starting a model:

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_local.json --dry-run
```

Preview an input-length by output-length by global-batch decode grid:

```bash
./tools/perf_test --mode=decode --action=run --config=rtp_llm/test/perf_test/examples/decode_grid_local.json --dry-run
```

Remove `--dry-run` to execute. Local prefill supports run, resume, profile,
and offline report; local decode supports run. Model and tokenizer paths must
already be available in the runtime. Public runs accept `execution.backend=local`
and require no internal checkout or private Bazel configuration.

See `rtp_llm/test/perf_test/docs/reference.md` for the capability matrix,
run-directory layout, result contract and exit codes. Mode-specific examples
are in the adjacent `docs/` and `examples/` directories.

## Multi-Node Benchmark

The multi-node launcher starts and stops the distributed service, then executes
the same normalized prefill or decode plan and result contract as the local entry. Only the
master rank measures cases and writes the final run; other ranks coordinate
service lifetime. Each host needs a working RTP-LLM environment and passwordless
SSH access from the launch host.

### Benchmark configuration

Use `rtp_llm/test/perf_test/multi_node/multi_benchmark_config.yaml` for decode,
or `rtp_llm/test/perf_test/multi_node/prefill_ssh_example.yaml` for prefill.
Both contain these sections:

| Section | Purpose |
| --- | --- |
| `experiment_name` | Names the experiment output directory. |
| `test_config` | Configures retries, build policy and diagnostic collection. |
| `machine_config` | Lists host addresses, SSH user and port. |
| `build_config` | Selects repository, revision, remote directory and build arguments. |
| `common_config` | Supplies common service settings. |
| `benchmarks` | Defines benchmark-specific service settings and a normalized prefill or decode JSON config. |

Every benchmark requires `fixed_config.perf_test_config`. This path is resolved
relative to the YAML file. The JSON config owns the workload, engine limits and model
topology. The launcher validates each expanded task before building or
contacting hosts. Its TP/DP/PP/EP settings must match the JSON config exactly.

For example, create a decode JSON config based on
`rtp_llm/test/perf_test/examples/decode_grid_local.json`, set its model and
tokenizer paths to files available on each host, and declare the topology and
devices for the distributed service. A two-worker DP config can be referenced
by a launcher benchmark like this:

```yaml
benchmarks:
  - benchmark_name: "Decode_DP2"
    fixed_config:
      perf_test_config: "./configs/decode_dp2.json"
      checkpoint_path: "/models/Qwen3-8B"
      tokenizer_path: "/models/Qwen3-8B"
      model_type: "qwen_3"
      tp_size: 1
      dp_size: 2
      pp_size: 1
      ep_size: 1
      local_world_size: 2
```

Supply the machine, build and common service sections from the launcher reference
for your environment. In this example the JSON config must declare
`model.topology = {"tp": 1, "dp": 2, "pp": 1, "ep": 1}` and the corresponding
devices.

An optional `iterative_search_config` zips equal-length lists by index; an optional
`recursive_search_config` forms the Cartesian product of its values. These expand
service configurations. If a task changes topology, it must reference a JSON
config with that topology. Decode workload axes remain in the JSON config:
fixed workloads use `input_lengths`, `output_lengths` and
`batch: {"strategy": "fixed", "sizes": [...]}`; TPOT workloads use
`batch: {"strategy": "target_tpot", "max_size": ..., "candidate_step": ...,
"target_tpot_ms": ...}`.

For prefill, the JSON workload can define cache grids, named grids, explicit
request batches, random batches or a traffic distribution. The controller
freezes traffic cases before delivery so the remote hosts do not need its
source distribution file. Rank zero uses the same cache probe, measurements,
result envelope and report generator as the local prefill path. A multi-request
prefill batch requires DP=1. The prefill example uses two hosts with TP=2 and
DP=1; update the model paths, revision and hosts for your environment.

The prefill example checks out the public repository directly and installs
requirements from its `deps/` directory. It does not require `internal_source`.
SSH, rank and collection failures return nonzero; collected run artifacts and
task logs remain available for diagnosis.

Run from the repository root after preparing your YAML and JSON configs:

```bash
python -m rtp_llm.test.perf_test.multi_node.multi_benchmark \
  -c rtp_llm/test/perf_test/multi_node/multi_benchmark_config.yaml
```

### Results and profiling

Decode batches are global across all DP workers. A global batch of 5 with DP=2
allocates 3 requests to one worker and 2 to the other. Results record
`global_batch_size` and each worker's local allocation. For target-TPOT search,
top-level metrics describe the selected best global batch; the configured
maximum and every attempted candidate remain in `metrics.search`.

Each case retains its requested output length, batch policy, status, request
outcomes, latency and throughput. Failed cases without formal metrics remain
diagnostic. A partial performance run has a nonzero exit code. An inline profile
failure preserves valid formal metrics and increments `profiling_failed_cases`.

Collected artifacts are stored under a benchmark's task directory:

```text
test_output/
└── Experiment_<name>_<timestamp>/
    └── Benchmark_<name>_<timestamp>/
        └── Task_hack_layer_num-4_dp_size-2/
            ├── test.log
            ├── copy.log
            ├── clean.log
            ├── process_logs/
            ├── trace_files/
            │   └── <manifest-listed-trace>.json
            └── run/
                ├── manifest.json
                ├── execution_plan.json
                ├── source_config.json
                ├── resolved_config.json
                ├── results/
                │   ├── result.json
                │   ├── summary.json
                │   └── cases/<case_id>.json
                ├── reports/
                │   └── decode_report.json
                └── profiles/
                    └── <case_id>/profile_manifest.json
```

`run/results/result.json` is the schema-v2 common envelope, containing run
identity, fingerprints, metadata, summary and ordered cases.
`run/reports/decode_report.json` contains flat observations and input-by-output
matrices grouped by exact batch or TPOT strategy. Profile manifests record
profile status, full input shape, requested output length, selected batch and
listed artifacts. Profiling must be enabled in the workload to produce them.
The collector always copies standard run artifacts and also preserves available
diagnostics for partial runs; optional process logs/traces depend on diagnostic
collection settings.

To merge results, pass the benchmark directory containing `Task_*` children:

```bash
python -m rtp_llm.test.perf_test.multi_node.analyzer.merge_decode_results \
  /path/to/Benchmark_run -o decode_results.csv
```

The merger also accepts standard run artifacts directly beneath `Task_*`.
It validates data with the shared result schema, diagnoses invalid files/cases,
and avoids counting the same run/case twice if direct and nested copies coexist.
CSV rows retain input/output lengths, strategy, global/selected batch, success
rate, latency, TPOT, throughput and status.

Analyze collected traces with:

```bash
python -m rtp_llm.test.perf_test.multi_node.analyzer.batch_analyze \
  /path/to/Benchmark_run --min-occurrences 20
```

Trace discovery reads completed profile manifests, takes DP from each task name,
and obtains shape and artifact paths from the manifest. Different output lengths
remain separate in every CSV/table. Failed manifests, missing artifacts and
nonuniform input distributions that cannot fit the scalar input-length axis
produce diagnostics.
