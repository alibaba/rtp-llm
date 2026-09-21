> Current traffic contract: see [traffic-sources-consolidation.md](../concepts/traffic-sources-consolidation.md).
> The source names/CLI recipes below describe the pre-migration experiment.
> New runs use `synthetic/realistic/1` or `trace/prefix_lineage/2`; pacing is in
> `client.playback`. Do not reuse old generated plans as a new-version baseline.

# P scale-in cache gate

This workload runs the existing Java load client, Master and Java mock engine.
It does not use Whale, a VIP server, GPUs or KMonitor. Python prepares a token
plan and controls phases; Java owns sending and Fetch/GenerateStream consumption.

Run the production-scale candidate with the existing parent runner. The remote
lease must contain the complete explicit window (example: Master at base, mock
at base+10, worker capacity 700 needs through base+712):

```sh
export FLEXLB_FT_WORKER_PORT_CAPACITY=700
python3 parallel_runner.py --parallel 1 --mock-stride 704 --profile single-nonbatch \
  --case-dir scale_cases/cache_scale_in_online.yaml --suite workload \
  --out-dir "$FLEXLB_RUN_DIR/cache-scale-in"
```

Set `FLEXLB_FT_PARALLEL_MASTER_BASE` and `FLEXLB_FT_PARALLEL_MOCK_BASE` inside the
allocated port block before execution. Java build and load run remotely; do not
start this workload on a laptop. The input YAML is a **calibration candidate**,
not a validated production capacity threshold or a historical-master A/B.

## Flow and verdict

The Java sender runs continuously. Two adjacent baseline half-windows must have
sufficient completed prefills, warm and stable token reuse, and target QPS before
scale-in is allowed. All selected P removals are submitted concurrently: each
withdraws discovery before draining, so drain periods do not serialize. The
post window starts when Master reports the target P count. Drain/topology budgets
must exceed the Master status-staleness interval; a shorter drain would cut
engines while they are still eligible for routing. This is checked at compilation. Discovery, graceful
drain, sampling and client pacing failures cannot produce PASS.

All gate parameters are explicit YAML data. `window_s` is the rolling hit window,
`step_s` its stride, `sustain_s` the minimum elapsed time between first and last
consecutive low observations. This deliberately does not count an entire rolling
window as sustained time from its first low point. The floor is
`max(absolute_min_hit, baseline_hit - max_drop)`; `max_drop` is a fraction, not a
relative percentage. Sustained collapse fails even if it later recovers. Short
dips and recovery are preserved in the curves.

Token reuse pools per-engine counter deltas over surviving engines:
`sum(delta hit_tokens_total) / sum(delta context_tokens_total)`.
`context_requests_total` counts at the same prefill completion point; the older
`completed` snapshot field does not count every NON_BATCH prefill path and must
not be used as this denominator. Removed engines' final counters are not
subtracted from fleet totals. Resets and incarnation changes invalidate windows.
Final offered QPS is reconstructed from the complete Java issued journal using
`send_start_epoch_ms` at each sample epoch. Live journal-consumption counts are
retained as `journal_observed_started`: buffered reads can otherwise create false
short-window rate spikes. This correction does not relax the QPS tolerance,
pacing limit, completeness or sampling requirements.

Waiting is diagnostic: an explicit minimum backlog establishes that the scenario
exercised its load precondition, but backlog size never independently fails cache
stability. No completed prefills means missing evidence, not a healthy cache.

- PASS: valid load/topology/evidence, no sustained low reuse.
- FAIL: valid evidence with sustained low reuse.
- INVALID: underexercised load, insufficient completions, off-target QPS, pacing
  lag, unstable topology, missing metrics, failed drain or incomplete client data.
  The scenario maps this to ERROR and exits nonzero. High rejection by itself is
  not FAIL; rejection that leaves too few completed prefills is INVALID.

Freeze thresholds only after known-bad/known-good controls. Shrink reduces total
cache capacity and causes legitimate cold loss; calibrate the floor against a
healthy run at the target scale. Passing 1P is not evidence that intermediate
scales are safe: 1P removes cross-P rerouting. Keep multi-P as the primary case.
The prior online master commits use config schema v1; this framework uses v3.
A historical comparison needs an explicit compatible adapter, not a blind JAR swap.

The candidate uses an explicit linear formula, shared by Master prediction and
mock execution: `12 + 0.025 * sum(computeTokens) + 0.001 * sum(hitCacheTokens)` ms.
It is a mechanism-test model, not the production DSv4 fit. It preserves a miss
penalty and has bounded service capacity even when the engine forms larger
batches. The earlier DSv4-formula 8P -> 4P smoke had only a shallow waiting queue;
that run validates orchestration, not severe overload. `prefill_expression` is an
optional common ConfigOverride; other scenarios retain their original formulas.

## Reproducible synthetic traffic

`synthetic/prefix_working_set/1` is registered in the shared traffic-source layer.
Parameters control seed, QPS, family count/skew, shared/private prefix depth,
cold fraction, unique suffix, session length/growth and output tokens. Arrival
rate and working-set breadth are independent. Actual tokens are materialized at
runtime; Java computes/validates cache hashes. Only generator/config belong in
Git. Logical cache-key block size is explicit (power of two, 64..4096); Java hashes
and advertises the per-record size. The production-scale trace uses 512 to match P.
Looping is forbidden because it turns unique cold suffixes into repeated hits.
The plan and duration must cover worst-case warmup, topology and observation.

The older `traffic_model` experimental prototype is in stash object
`163252155c47d2c03e20c31b8567674b169eb98e`, third parent, under
`rtp_llm/flexlb/tools/online_eval/traffic_model/`. It has a richer fitted trie and
session model, but emits block hashes without canonical input tokens. The earlier
three-family implementation is commit `1450a74ef7`. Neither is assumed to be a
validated production distribution here.

Inspect generated traffic without KMonitor:

```sh
python3 -m online_eval.workload_profile PATH/flows/scale_in.jsonl \
  --output traffic-profile.json --capacities 128 256 512 1024
```

This reports shared-prefix potential, distinct blocks, family frequencies,
repeat-gap quantiles and ideal fully associative LRU capacity curves. It does not
simulate multi-P affinity, tree eviction, pinning or H2D visibility. Synthetic
realism remains `NOT_VALIDATED` until representative traces validate these traits
and the intended regression is reproduced under controlled traffic.

## Artifacts and second review

The workload report bundle links `reports/run/cache-scale-in/report.html`; gate analysis lives alongside it in `analysis.json`.
`cache-gate-evidence.json` contains parameters, counter samples, actual events and
completion-time curves; `cache-gate-result.json` records every adjudicated window.
`cache-gate-samples.jsonl` is incrementally flushed so interrupted runs retain
observations. Original request journals and engine events remain in the existing
framework artifact tree. Missing observations remain gaps.

The page shows hit rate/floor, actual P, waiting/running, sent/terminal traffic,
success/failure/prefill completion QPS, Device/Memory reuse, executed context batch
size and successful TTFT. Phase markers use observed timestamps, not scheduled
ones. Forward means and request-weighted batch sizes are labeled explicitly.

Recompute the identical verdict offline without acquiring fresh metrics:

```sh
python3 -m flexlb_test_framework.workload.cache_gate PATH/cache-gate-evidence.json \
  --output replayed-report
```

The gate uses the common renderer and shared workload telemetry. It does not
change master scheduling code. The mock adds completion/reuse evidence only.

## Initial execution validation (2026-09-21)

The candidate completed a remote Java run: 80 QPS, 8P/8D -> 4P/8D,
180 seconds of post-convergence observation, 343 seconds total runner duration.
Gate PASS and workload VALID; baseline token hit was 70.57%, rolling post-shrink
hit 66.60–72.26%, waiting peak 232. All started requests had terminal accounting
(including failures); all four P drained and cleanup passed.

The Java build and 28 Java parity/control tests passed. Python full discovery ran
939 tests with five report-class setup errors caused by the legacy renderer
symlink; after fixing resource resolution, all 63 affected report and gate/client/
traffic tests passed. The slow full suite was not repeated. Offline verdict
recomputation matched the remote result. These results validate a healthy
overloaded scenario, not a historical known-bad/known-good regression control.

## Production-scale candidate and shared overlay component

`config/scale_cases/cache_scale_in_online.yaml` uses 125P/536D, fixed 240 QPS and one-step
125P -> 24P. Historical real traffic in the old-master comparison had median
235.79 QPS (range 207.38–282.09 QPS across 231 buckets); 240 is a fixed reference,
not a live reading. Decode remains at 536 throughout. The original small case
remains in the default catalog for orchestration smoke only.

The explicit larger worker-port capacity relocates victim/control tail ports
from offsets 149..151 to 700..702. Parent probes, numeric interval locks, compiler,
lease validation and harness use the same bound. Default capacity stays 149.
Do not increase P/D without allocating the larger remote lease and setting this
variable in the parent process before compilation. Overlapping lanes are rejected.

Performance preset `production_scale_20260920` restores the archived fitted
Prefill formula (`77.82664607924481 + 112.5170362714666 * batchSize +
10.87594301934867 * sum(computeTokens / 1024.)`) in both Master prediction and mock
execution. Decode uses 33.4254796 + 0.27851927 * running ms/step and 2.8638118
tokens/step, EOS mean 420. Per P: Device 31,218 blocks of 512 tokens; Memory 95,311
blocks, flat LRU, H2D consume lifecycle. Per D: 46,157 blocks of 64 tokens.
Device tree is enabled to match the stated real policy; the early archived mock
snapshot had it disabled. This is an explicit correction, not a claim that the
archive already contained the correct value.

The synthetic input remains 16–22 logical 512-token blocks with 1,024 families,
Zipf 0.9, 5% cold inputs, four-request growing sessions. It is NOT_VALIDATED against
production length/prefix/reuse-distance distributions; matching QPS and P/D does
not establish equivalent pressure or the same critical P. Generated traces stay
outside Git. Large-run warmup and Java loading are bounded separately.

`stress/multi_curve.js` is a reusable panel mounted by the shared renderer when
`panel.overlay=true`. Its input contract is `series[{name,points,axis,unit,color,
hidden}]`, `axes{axis:{title,position,min,max}}`, and named `presets`. Curves retain
independent sampling timestamps, original units and missing-value gaps. The
report defaults to hit/P/waiting/sent/success/failure on one timeline; toggles,
unit axes, preset selections and time bounds support follow-up inspection.

The first large-scale attempt exposed a client/engine key-block mismatch: the
legacy client always hashed/advertised 1024 while P used 512. That attempt was
stopped during warmup and is excluded from gate evidence. Per-record block size
is now validated, hashed and preserved on schedule requests and trace truncation;
legacy records without this field retain 1024. The Python canonical-plan validator
uses the same supported-size set. Remote client/control tests cover this contract.

Large flows derive their lifecycle-event budget from the validated canonical
manifest: two events per request, at least the legacy 50,000 allowance and never
more than 2,000,000. Incremental read-size limits and event-sequence validation
remain active. This avoids silently truncating the observation around request
25,000 while preserving a hard memory/evidence bound.

## Production-scale execution result (2026-09-21)

Remote 134 ran 125P/536D -> 24P/536D at 240 QPS, observing the full 180 seconds.
Baseline reuse was 67.11%; post-shrink valid ratios 23.34–48.56%, waiting peak
2,413. The gate returned INVALID because eight overlapping windows lacked the
minimum 20 completed prefills (three had none); 142 seconds of low hit were
recorded outside those gaps. Device eviction deltas were zero, and many client
failures were 10-second deadlines. Do not call this an eviction-storm proof.
All 101 P drained, client terminal accounting completed and cleanup passed.
Input preparation/loading took 206 s; warmup/scale/observation 235 s; the complete
job including generic reports took about 746 s. Known-bad/known-good calibration
and faster preparation/reporting remain pending before enforcing this in CI.

## Historical Whale Master controls

`config/scale_cases/cache_scale_in_whale_ab.yaml` reproduces the archived configuration
with 125P/536D -> 8P/536D, 240 QPS, NON_BATCH, flat Device / flat Memory eviction
and consume-on-H2D. This intentionally differs from the current production-scale
candidate's Device tree setting. It is the historical mock configuration, not a
claim that the archived mock matched real Device eviction. The Java client RPC
deadline is 300 seconds; the 44P pilot did not reproduce the old regression. Synthetic prefix realism remains unvalidated.

Set `FLEXLB_FT_HISTORICAL_MASTER_MANIFEST` to an absolute JSON manifest containing
`source_commit` (40-character SHA), `jar`, `jar_sha256`, `config`, `config_sha256`.
The adapter verifies both files before launching; config must be schema-v1
NON_BATCH. Build both commits remotely under opensource with the same opt-in
`tools/whale_mock/discovery_adapter/WhaleFileDiscovery.java` Bean. Production
Java source remains unchanged; no VIP or KMonitor is required. The adapter only
changes the Master artifact, config and file-discovery environment; the scenario
still owns mock execution timing. `actual-master-config.json` and the manifest
are archived separately from `master_config.json` (mock formula envelope).

Run both versions sequentially with the same YAML and distinct output directories.
Then `python -m flexlb_test_framework.workload.cache_gate_ab OLD_EVIDENCE NEW_EVIDENCE
--output AB_DIRECTORY` checks criteria, topology, performance, actual Master
config, mock formula and trace SHA equality, preserves individual decisions and
renders two multi-curve panels aligned at withdrawal. Only aligned FAIL/PASS
controls satisfy `expected_control_observed`; this does not assert production
realism or prove an eviction-storm cause.

For the overloaded historical controls, give the Java stream RPC a deadline
longer than the 180-second observation (the paired validation uses 300 seconds)
and leave client concurrency headroom (32768). A 60-second client deadline caused
mass cancellations and insufficient completed-prefill windows on the new Master;
that run remains INVALID, even though hit rate recovered. The check thresholds
are unchanged. Drain timeout scales with the explicit client deadline, and the
scenario cleanup budget must cover it.

To avoid regenerating the same large synthetic plan, the source registry supports
`trace/canonical/1`: parameters `path`, `sha256`, `count`, and
`identity: preserve_namespaced`. It copies exact bytes only after SHA verification,
then validates canonical records, count and the caller's namespace. Use a runtime
artifact generated by the same synthetic source; do not commit the large trace.
The manifest continues to identify its source, and A/B compares the actual trace
SHA as well as normalized Java client settings. This optimization changes neither
tokens nor arrivals.

### Frontend-derived empirical prefix model

`trace/prefix_lineage/1` consumes a SHA-pinned gzip JSON model (`version: 1`,
`block_size: 512`, `events: [[original_ms, input_length, parent_index, shared_blocks], ...]`).
Parameters are `path`, `sha256`, `count`, `qps`, `output_tokens`, and `priority`.
References must point backward and cannot exceed either request's complete blocks.
It preserves measured complete-block prefix relationships and input lengths using
anonymous token labels. It is a finite empirical model, not a statistical fit with
validated generalization; do not loop it or claim sub-block text reconstruction.
The original arrival ordering is retained; explicit fixed QPS changes time gaps.

For large plans, `input_token_blocks` is a compact alternative to `input_ids`:
each integer is repeated `cache_key_block_size` times, with the final block cut at
`il`. Exactly one encoding is accepted. Python validates shape and Java calculates
cache hashes from the expanded sequence, including checking supplied keys. Java
keeps the token sequence lazy to avoid storing billions of boxed repeated integers.
RPCs still carry the expanded tokens. `FLEXLB_FT_MOCK_HEAP` can override the normal
2 GB mock heap for long-input, deep-queue experiments; use identical heaps in A/B.

Collect the same interval across all frontend shards with a common prefix hash,
then merge before fitting. Record missing pods and log coverage. Error-censored
output lengths cannot establish normal decode demand; choose an independently
calibrated output model explicitly. Keep captures/model artifacts outside Git.
Passing a threshold gate alone does not establish reproduction of the historical
90%-to-teens sustained collapse; report baseline, tail recovery and evictions.

Pod-local capture CLI: `python3 -m online_eval.capture_frontend_prefix --start EPOCH_MS --end EPOCH_MS --out /tmp/OWNED_PREFIX` from the frontend working directory. It reads the existing access logs at low priority when launched with `nice`, writes compressed prefix metadata and a coverage/SHA summary, and does not alter service configuration. Collect the same interval on each pod and name the artifacts `pod-N.jsonl.gz` and `pod-N.summary.json`.

On the development host, run `python3 -m online_eval.fit_frontend_prefix --source CAPTURE_DIR --out FIT_DIR --qps 240 --expected-pods 20 --output-tokens 420`. Choose the output cap explicitly from the calibrated model; these numbers are an example, not universal defaults. The fit emits an empirical lineage model, compact plan and provenance report. Missing pods remain explicit and mismatched capture intervals are rejected.
