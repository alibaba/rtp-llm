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
python3 scripts/commands/run_cases.py --parallel 1 --mock-stride 704 --profile single-nonbatch \
  --case-dir scale_cases/cache_scale_in_lineage.yaml --suite workload \
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
The prior online Master commits use config schema v1; the default framework
uses v3. The ordinary launcher accepts an explicit config file and adapts
schema-v1 file discovery at startup; both inputs are recorded in evidence.

The small orchestration smoke uses an explicit mechanism-test formula:
`12 + 0.025 * sum(computeTokens) + 0.001 * sum(hitCacheTokens)` ms. The
production-scale lineage case below uses the archived DSv4 fit. The 8P → 4P
smoke validates orchestration, not severe overload. `prefill_expression` is
an optional common ConfigOverride; other scenarios retain their formulas.

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
python3 -m traffic.workload_profile PATH/flows/scale_in.jsonl \
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
python3 -m workload.cache_gate PATH/cache-gate-evidence.json \
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

`config/scale_cases/cache_scale_in_lineage.yaml` uses 125P/536D, fixed 240 QPS and one-step
125P -> 64P. Historical real traffic in the old-master comparison had median
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
The retained lineage control uses flat Device eviction, matching the archived
mock A/B setup rather than the stated real Device tree policy. This is a
controlled reproduction setup, not a claim that the archived mock matched
the real Device policy.

The retained input is a finite, SHA-pinned empirical frontend prefix-lineage
model, not the retired synthetic-family generator. It preserves observed input
length and complete-block prefix relationships; output length remains the
calibrated 420-token value. Matching QPS and P/D alone does not establish the
same critical P. Generated request traces stay outside Git.

`src/reporting/assets/multi_curve.js` is a reusable panel mounted by the shared renderer when
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

## 单 run 与下游 A/B 分析

`config/scale_cases/cache_scale_in_lineage.yaml` 是真实前端前缀谱系流量的
125P/536D → 64P/536D 单 run 门禁。它使用普通 Master 启动路径，
判定仍是原有绝对命中率、完成量和有效性规则。该配置保留已知 A/B
分离点的负载和容量；正式门禁前仍须核验两侧观测结果。

默认运行使用本地构建的 Master JAR。测试另一版本时，可分别设置
`FLEXLB_FT_MASTER_JAR` 为其 JAR 路径、
`FLEXLB_FT_MASTER_CONFIG_FILE` 为其实际配置文件路径；
schema-1 配置在 discovery_file 场景下自动接入文件发现。
`FLEXLB_FT_MASTER_SOURCE_COMMIT` 可选，仅用于声明构建来源；
JAR 内有可识别的 commit 字段时会自动读取。声明值与实际 JAR
哈希分开标注，不能把声明当成已验证源码。每轮 evidence 的
`provenance.master_artifact` 记录实际 JAR 路径、SHA256、
source commit 及其来源；`actual_master_config` 记录实际生效配置。
不提供 commit 仍可运行，报告会明确显示缺失，JAR 哈希始终可追溯。

两侧依次使用同一 lineage YAML，分别输出独立 run 目录。
`config/scale_cases/cache_scale_in_historical_ab.yaml` 只定义下游分析
策略，不参与启动或编排：

```sh
PYTHONPATH=src:. python3 -m workload.cache_gate_ab OLD_RUN_DIR NEW_RUN_DIR \
  --config config/scale_cases/cache_scale_in_historical_ab.yaml --output AB_DIR
```

对比层检查流量 SHA、拓扑、容量、性能公式、Master 实际配置和
Java client 设置；缺失控制字段显示 UNKNOWN，不会算作一致。
两侧监控曲线按 withdraw_start 对齐，首页同图叠加关键曲线，
并保留单侧详情。默认 strong 模式仅在控制变量一致且 old FAIL /
new PASS 时输出 CONTROL_OBSERVED。传 `--mode weak` 仅要求控制
变量一致，传 `--mode none` 仅出报告并照常列出差异或缺失。
对比不改变任何单 run 的 PASS / FAIL / INVALID 判定。

远端 125P 测试仍需大于观测期的 Java stream RPC deadline 和足够的
并发余量。若基线阶段大量取消或完成量不足，单 run 会返回 INVALID，
不能当成回归对照。

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

Pod-local capture CLI: `python3 -m traffic.capture_frontend_prefix --start EPOCH_MS --end EPOCH_MS --out /tmp/OWNED_PREFIX` from the frontend working directory. It reads the existing access logs at low priority when launched with `nice`, writes compressed prefix metadata and a coverage/SHA summary, and does not alter service configuration. Collect the same interval on each pod and name the artifacts `pod-N.jsonl.gz` and `pod-N.summary.json`.

On the development host, run `python3 -m traffic.fit_frontend_prefix --source CAPTURE_DIR --out FIT_DIR --qps 240 --expected-pods 20 --output-tokens 420`. Choose the output cap explicitly from the calibrated model; these numbers are an example, not universal defaults. The fit emits an empirical lineage model, compact plan and provenance report. Missing pods remain explicit and mismatched capture intervals are rejected.
