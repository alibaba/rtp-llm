# Local decode performance tests

Decode supports local `run` only. A run starts one engine session, builds prompts
for the distinct planned input lengths, and executes all cases through the
unified batch and case executors. The run writes the common result, summary, and
`reports/decode_report.json`.

## Fixed grid

An exact grid is the Cartesian product of positive `input_lengths`, positive
`output_lengths`, and `batch.sizes`, with `batch.strategy` set to `fixed`. The
example has three inputs, two outputs, and three batch sizes, producing 18 cases.
Planned output lengths override `min_new_tokens` and `max_new_tokens` in
`generate_config`.

With DP greater than one, every global batch is deterministically divided across
workers and the result records both the global batch and worker-local allocation.
Uneven batches are supported. The report contains one input-by-output matrix per
exact batch size, retaining status, TPOT, throughput, total latency, and success
rate for every cell. Failed or incomplete cases produce a diagnostic report.
Persisted rounds include their index, worker allocations, throughput, and one
outcome per request. Readers validate request shapes and timings and reconcile
the aggregate metrics with those rounds before including a case in formal reports.

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_grid_local.json \
  --dry-run
```

## Distribution

The submit process reads and validates the small distribution file, then embeds
its content in the resolved workload. The remote source path is not retained.
Distribution workloads retain one positive scalar `output_length` and use the
same explicit `batch` policy as grids. `sequence_lengths` must contain enough
positive lengths for the largest global batch. The example uses exact batches
2 and 5, producing two cases. Each result preserves the actual lengths and all
request outcomes. Distribution reports contain ordered observations rather than
a rectangular matrix. A per-batch length map supports exact batches only.

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_distribution_local.json \
  --dry-run
```

## Target TPOT

Set `batch.strategy` to `target_tpot` and supply `max_size`, `candidate_step`,
and `target_tpot_ms` instead of `sizes`. Each input/output pair independently
searches for the largest usable global batch that meets the target. The example
has three inputs and two outputs, producing six search cases, with maximum batch
32, candidate step 4, and target TPOT 10 ms.

Every attempted batch, capacity or measurement failure, TPOT value, success rate,
and final selection is retained in the search trajectory. Top-level performance
metrics describe the selected best batch. Each input-by-output report cell shows
the search outcome, best batch, actual TPOT, and throughput. If no candidate
satisfies the target, the case records `target_tpot_not_met` and the run is partial
with a nonzero exit code.

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_tpot_local.json \
  --dry-run
```

## Inline profiling

Profiling is disabled by default. Enable it in the workload to profile each
successful case after formal measurement within the same `run` session:

```json
"profiling": {
  "enabled": true,
  "runs": 1,
  "steps": 3,
  "enable_all_ranks": true,
  "flush_timeout_seconds": 60
}
```

These are the default profile settings apart from `enabled`. Exact cases profile
their measured batch; target-TPOT cases profile only the selected best batch.
Profile requests and timing stay separate from formal measurements. Each attempt
writes `profiles/<case_id>/profile_manifest.json` with the shape, settings, trace
identity, discovered artifacts, and outcome. A profile failure preserves valid
performance metrics and increments the summary's `profiling_failed_cases` count.
Both log-marker discovery and filesystem discovery accept complete JSON exports
with a `traceEvents` array and retain them atomically under `traces/` in the run.
Multi-node collection copies those retained files with the run, and trace analysis
resolves remote manifest paths to the collected `Task_*/run/traces/` files.

Decode `resume`, `profile`, and `report` are unsupported and return
configuration exit code 2. The public CLI accepts only local execution.
Manual SSH prefill/decode uses the controller documented in the benchmark guide.
