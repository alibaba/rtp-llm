# Local prefill performance tests

The unified prefill mode measures server first-token latency while controlling
the exact cache reuse and compute lengths. It accepts only the versioned unified
configuration format. Update the checkpoint, tokenizer, topology, and device
fields in `examples/prefill_local.json` for the target environment.

## Measurement semantics

Local and manual SSH execution use the same prefill runner. Cached cases first
require acknowledgement of prefill scheduler batch size 1, then seed each member
serially. After all seeds pass validation, the runner requires acknowledgement of
the target batch size B before measurement. Cold cases configure B directly.
Scheduler or seed failure stops the case before any measured requests are sent.

Each measured round submits B requests to a reusable B-worker pool and releases
them with a common signal. The runner waits for every member to finish, including
when one fails. Any failed request, invalid shape, incorrect reuse, or invalid
timing fails the whole case and leaves its metrics empty.

Throughput uses the complete round's monotonic wall-clock duration, starting at
the common release and ending after all requests finish. Prompt construction,
seeding, and scheduler updates are outside this interval. Each round contributes
the total computed tokens divided by `round_elapsed_seconds`; the case reports
the arithmetic mean of these round rates, including for B=1. Server first-token
latency is a separate metric: the case records the median across measured member
responses, while the report's batch latency is the median of each round's maximum
member latency. For example, B=4 and compute_len=8 over a 0.2-second round gives
160 tokens/s regardless of individual first-token latencies of 10–40 ms.

Flattened member evidence records the run, member index, server latency, observed
reuse length, shared `round_elapsed_ms`, and measurement contract
`concurrent_batch_v1`. Explicit request groups use
`concurrent_grouped_batch_v1` and include each member's input, reuse, and compute
lengths. All member latencies and round durations must be finite and positive.

## Grouped and ordered grids

`examples/prefill_grouped_local.json` shows `request_batches`, where every batch
has `request_groups` entries with `count`, `input_len`, and `cache_len`. Counts sum
to the measured batch size. `workload.workspace_tokens` is required; planning
checks raw and block-padded token budgets and every member's sequence length.
The runner seeds cached members independently before measuring them together.

`examples/prefill_multi_grid_local.json` shows `workload.grids`. Grid names are
unique and execution follows their declared order in one local engine session.
Each grid has generated cases or explicit `request_batches`; shared alignment,
commit tail, measurement runs, and workspace settings stay at workload level.
The merged report keeps each case's grid name.

Preview either example before running it:

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_grouped_local.json --dry-run
```

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_multi_grid_local.json --dry-run
```

## Random heterogeneous batches

`examples/prefill_random_batch_local.json` declares `random_request_batches`
in the unified workload. Preview or run it with the same entry:

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_random_batch_local.json --dry-run
```

The random object accepts `num_cases` (default 100), `batch_sizes` (default
`[2, 4, 8]`), `min_input_tokens` (8192), `max_input_tokens` (262144),
`input_alignment` (256), and `cold_probability` (0.2). Optional
`batch_input_limits` maps batch-size strings to per-member input limits, for
example `{"4": 6144}`. Shared `seed` defaults to 20260917 in random mode.
`workspace_tokens` is required. This declaration replaces explicit
`request_batches` or flat generated-grid fields; one named grid may also use it.
Shared cache alignment, tail, measurements, seed, and workspace settings remain
at workload level.

A fixed config and engine produce deterministic, distinct request distributions.
Every requested batch size appears, so `num_cases` must be at least the number
of sizes. Batches larger than one contain at least two different input/cache
geometries. Each member has one request group; member order is deterministically
shuffled. Inputs respect alignment, per-size limits, sequence length, and the
one-token output reserve. Cache lengths are cold or block aligned and leave the
commit tail. Sampling checks raw batch tokens and block-padded workspace, then
the normal grouped planner admits the cases. Optional `kv_budget_tokens` bounds
the conservative peak estimate of padded active inputs plus one commit tail for
each cached member. This sampling bound does not guarantee against OOM.
Impossible bounds or insufficient distinct shapes fail before model launch.

Freeze the mixed sampled workload as a full unified config:

```shell
python3 -m rtp_llm.test.perf_test.prefill.generate_random_batch_grid \
  --config rtp_llm/test/perf_test/examples/prefill_random_batch_local.json \
  --output /tmp/prefill-random-frozen.json
```

`--output` supports named grids and preserves their names and declared order,
including explicit or generated siblings. The export stores ordinary
`request_batches` and the effective seed, preserving grouped planner cases.
For a single top-level random workload, export one file per batch size:

```shell
python3 -m rtp_llm.test.perf_test.prefill.generate_random_batch_grid \
  --config rtp_llm/test/perf_test/examples/prefill_random_batch_local.json \
  --output-dir /tmp/prefill-random-batches
```

Files are named `batch_2.json`, `batch_4.json`, etc. Each contains `num_cases`
cases for that size using `seed + batch_size`; it resamples that size rather
than filtering the mixed export. Both modes refuse to overwrite destinations,
and directory export checks all destination collisions before writing any file.
Exactly one output option is required. Exported files include model, engine,
execution, and output settings and run through the unified entry:

```shell
./tools/perf_test --mode=prefill --action=run --config=/tmp/prefill-random-frozen.json --dry-run
```

Remove `--dry-run` to measure. Public configs use `execution.backend=local`
as in the local examples. Random sampling and exported groups use the same
concurrent grouped runner for local runs and manual SSH multi-host runs.

## Traffic-sampled batches

`examples/prefill_traffic_distribution_local.json` samples correlated input and
reuse lengths from the adjacent `req_distribution.json`. Update its model,
tokenizer, device and budgets for your environment. Its explicit 512-token
minimum excludes short prompts that may not support synthetic construction.
Preview with:

```bash
./tools/perf_test --mode=prefill --action=run --config=rtp_llm/test/perf_test/examples/prefill_traffic_distribution_local.json --dry-run
```

`traffic_request_batches` replaces explicit or random batches, at workload
level or in a named grid. `distribution_file` is relative to the config;
`num_cases` is the total, with `batch_sizes` assigned cyclically. Specify
`min_input_tokens`, optionally `max_input_tokens`, `kv_budget_tokens`, and
`retry_limit`. Shared workload settings control seed, cache alignment,
commit tail, workspace and measurements. Inputs retain sampled lengths; reuse
is floored to cache blocks and clamped to leave the commit tail. Whole batches
that exceed admission budgets are rejected and redrawn. Budget rejection alters
the accepted distribution. The resolved sampling summary records the source
digest, eligible fraction, rejection counts and accepted length statistics.
These are sampled geometries, not a replay of production arrivals or content.

Freeze the same cases as explicit batches with:

```shell
python3 -m rtp_llm.test.perf_test.prefill.generate_traffic_batch_grid \
  --config rtp_llm/test/perf_test/examples/prefill_traffic_distribution_local.json \
  --output /tmp/prefill-traffic-frozen.json
```

The exporter refuses to overwrite an existing file. Sampling
occurs on the controller; SSH delivery carries frozen cases and the summary,
and workers do not need the distribution file. The immutable execution plan
and compact `results/cases.jsonl` drive resume without resampling.

## Cache probe admission

Every engine session probes cache reuse before formal measurements. Planning
checks the seed and two-block continuation, each with one output token, against
`engine.max_seq_len`, `engine.max_batch_tokens`, and block-padded
`workload.workspace_tokens` when specified. These checks apply to generated
flat grids, named grids, and explicit groups.

The shared planner/runtime block expectation is `workload.probe_block_size`,
then `engine.args.seq_size_per_block`, then `workload.cache_alignment`.
Enabling `engine.args.prefill_cp_kv_cache_sharded` or
`engine.env.PREFILL_CP_KV_CACHE_SHARDED` requires an explicit
`workload.probe_block_size` before engine launch. Arguments override environment
settings. Choose the model's effective cache-key block size; multiplying the
base size by CP size is not universally valid. Group alignment and padded
workspace use this same value, and the live probe verifies actual reuse.

## Run

`run` is the only action that accepts `--config`. The preview validates and
normalizes the config, expands the cache grid, computes fingerprints, and shows
the local resource plan without creating a formal run directory or starting a
model.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/prefill_local.json \
  --dry-run
```

Remove `--dry-run` to execute the plan. The created run directory contains the
source and resolved configs, immutable execution plan, manifest, per-case
results, checkpoints, reports, profiles, and traces. A successful local or SSH
run generates `reports/prefill_report.json`, standalone interactive latency and
tokens-per-minute charts (`latency.interactive.html`, `tpm.interactive.html`),
SVG fallbacks (`latency.svg`, `tpm.svg`), legacy projections
(`prefill_3d.svg`, `prefill_cold_miss.svg`), and `reports/formula/fit_report.json`
in the same command. The fit report records rejection when there is insufficient
data or the validation gates fail; a formula text file appears only for a valid
candidate.

## Resume

`resume` accepts only a run directory created by the unified entry. It checks
the frozen config, workload, model, and runner fingerprints, preserves every
successful case, and reruns the remaining planned cases.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=resume \
  --run-dir=perf_test_results/prefill-run \
  --dry-run
```

Persisted prefill plans must mark every case with measurement contract
`concurrent_batch_v1` or `concurrent_grouped_batch_v1` to resume or profile,
including dry runs. Older directories remain readable with offline `report`,
but new measurements require a fresh
`run --config=...` in a new directory. Existing plans are never migrated in place.

## Profile

`profile` reconstructs selected original requests from their stable case IDs.
Its artifacts are isolated under `profiles/`; formal case results, checkpoints,
and summaries remain unchanged. Case IDs come from the run's execution plan.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=profile \
  --run-dir=perf_test_results/prefill-run \
  --cases=prefill-0123456789abcdef01234567 \
  --dry-run
```

## Report

`report` is offline. `--skip-test` selects the same prefill report action when
`--action` is omitted. It reads the immutable plan and committed case files,
excludes missing or invalid points, and writes a formal report only when all
planned cases are valid. Otherwise it writes a diagnostic report and exclusion
reasons. It does not launch a model or modify measured cases.

```bash
./tools/perf_test --mode=prefill --skip-test --run-dir=perf_test_results/prefill-run
```

```bash
./tools/perf_test \
  --mode=prefill \
  --action=report \
  --run-dir=perf_test_results/prefill-run \
  --dry-run
```

`--skip-test` requires prefill mode and `--run-dir`; it accepts an optional
`--action=report` and supports `--dry-run` preview. It cannot be combined with
run, resume, profile, or a config override. `resume`, `profile`, and `report`
never accept a config override. Local prefill supports all four actions.

## Contract labels and migration

Prefill plans containing any `batch_size > 1` require `model.topology.dp == 1`
on every backend. Plans containing only batch size 1 still support DP > 1; TP
and PP remain supported. The concurrent batch measures one engine, so
distributing members across independent DP engines would change its meaning.

JSON observations and the HTML report expose `measurement_contract`:
`concurrent_batch_v1` uses measured round wall time for throughput, while
`legacy_latency_derived_v1` labels historical latency-derived observations.
Offline reports still read old directories without rewriting plans or results.
New-contract reports validate member indices, reuse, shared round durations,
and aggregate latency and throughput against the persisted raw evidence.
Inconsistent cases are excluded and make the report diagnostic.

The contract marker participates in stable case IDs, so fresh runs have different
IDs even for the same lengths and batch size. Old baseline keys fail with missing
cases; they never silently match the new measurements. Result schema v2 remains
unchanged. Compare measurements only when contract labels match. Normal runs
and report generation never rewrite accepted baseline artifacts.

The standalone charts embed Plotly and provide 3D compute/cache/batch geometry,
batch selection, sum/mean lengths, case/round samples, and latency/input TPM/
computed TPM controls. Click a point for persisted measurement evidence. Without
Plotly, the HTML embeds its static SVG. All throughput uses measured round wall
time; historical latency-derived observations stay excluded. Offline `report` and
`--skip-test` regenerate all six chart files without launching the model or changing
plans, checkpoints, or case results.

Prefill case results are appended to `results/cases.jsonl`; the latest complete record for each case is used for resume and reporting. The fingerprint checkpoint remains fixed as cases finish. Runs using the older `results/cases/` layout must be started again.
