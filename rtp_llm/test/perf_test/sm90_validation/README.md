# SM90 and default SM100 validation

[matrix.json](matrix.json) lists 61 source-reviewed Bazel targets and their
coverage. Merge each row with `defaults`, its `lane`, and optional `env_profile`.
GPU count defaults to one; model, PD and optional-fixture exceptions are explicit.
Durations are unmeasured until the actual candidate runs. The matrix does not
certify a build or test result.

## Dependency lanes

| Lane | Bazel selection | DeepGEMM | CUTLASS DSL |
| --- | --- | --- | --- |
| SM90 x86 | `--config=cuda13 --repo_env=RTP_LLM_SM90_DEPS=1` | `2.8.2+8692811` | `4.8.1` |
| SM100/103 x86 | `--config=cuda13 --repo_env=RTP_LLM_SM90_DEPS=0` | `2.8.2+341b656` | `4.7.1` |
| SM100 ARM | `--config=cuda13_arm --repo_env=RTP_LLM_SM90_DEPS=0` | upstream ARM `341b656` | `4.7.1` |

Use distinct output roots for each worktree and dependency lane. Do not expand
both `cuda13` and `cuda13_sm90` in one command. Confirm the actual GPU capability,
selected pip repository, DeepGEMM metadata/import path and native libraries.
Pool labels and source checks do not replace these runtime facts.

Before each invocation, use the paired checkout's
`internal_source/.cursor/skills/test-execution` and `rtp-worktree-test` instructions:
verify a non-root container user, run `pre_build_check.sh`, reserve the appropriate
GPUs and preserve the canonical GPU lock. Remote execution remains opt-in. Fill
`CONTAINER`, `CONTAINER_USER`, `PUBLIC_REPO` and `OUTPUT_ROOT` with verified paths;
`SKILL` below is the paired checkout's test-execution skill directory.

## Core kernel regression

Run the matrix's required one-GPU targets first. For example:

```bash
"$SKILL/bazel_run.sh" "$CONTAINER" "$CONTAINER_USER" "$PUBLIC_REPO" \
  "$OUTPUT_ROOT" sm90_scale_commit \
  test //rtp_llm/models_py/modules/dsv41/test:test_sm90_quant_scale_mn_major \
  //rtp_llm/models_py/modules/dsv41/test:test_dspark_commit_wkv_sm90 \
  --config=cuda13 --repo_env=RTP_LLM_SM90_DEPS=1 \
  --test_env=RTP_LLM_SM90_NVIDIA_WHEEL_PATH=1 \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --test_output=errors --test_timeout=3000 --nocache_test_results
```

The wrapper adds the canonical GPU lock. Other required groups cover BF16 gate
routing/input packing, native G32 dense/shared scales, DSpARK prepare/commit,
584B cache/CP, paged indexer/TopK and Engram. Run default SM100/103 and ARM targets
with their own lane; do not enable the SM90 dependencies for those regressions.
Common C++ targets must run on actual hardware in both dependency lanes.

Two boundary contracts are separate:

- The two-kernel fused HC boundary supports at most 64 rows. Its existing graph
  tests cover changed inputs and deferred shared/routed combination. The new
  33/63/64 numerical, 64-row graph and 65-row fallback cases require a fresh run.
- The standalone SM90 prenorm fast path supports at most 256 rows. The focused
  tests cover 255/256 against the existing reference and changed-input capture;
  257 must return the existing fallback without probing GEMM, including capture.

`test_v41_prefill_global_mapped_sm90` runs the existing mapped compression/state
oracle on Hopper for 288B and 584B pools. The SM100 target retains its upstream
288B format. The tests check latent, cache bytes, state and carry without changing
production codecs or tolerances. `test_v41_sm90_v4_cache` also covers the CP8
fresh-store ordering, old-prefix reads and untouched padding.

## Full-model smoke and performance

Use the full 40-layer Flash checkpoint and its tokenizer. Official role targets
reserve eight GPUs: CP8 prefill or TP1/DP8 decode. Supply this lane's environment:

```text
--test_env=CHECKPOINT_PATH=/path/to/full40-checkpoint
--test_env=TOKENIZER_PATH=/path/to/full40-checkpoint
--test_env=SM90_PERF_MOE_STRATEGY=mega_sm90_se
--test_env=DSV41_ENGRAM_DEVICE_SCALES=0
--test_env=DSV41_CPU_ENGRAM_LOOKUP=0
--test_env=RTP_LLM_DSPARK_FORCE_TOPK=0
--test_env=RTP_LLM_SM90_NVIDIA_WHEEL_PATH=1
--test_env=TMPDIR=/tmp/short-owned-directory
```

The role recipe selects its strategy from `SM90_PERF_MOE_STRATEGY`; setting only
`DSV4_MOE_STRATEGY` does not select fused shared expert here. The explicit CPU
lookup setting validates the UVA/prefetch startup path. It is a test selection,
not an instruction to change a deployment.

| Stage | Exact target suffix under `//rtp_llm/test/` | Inputs and outputs |
| --- | --- | --- |
| P smoke | `smoke:deepseek_v41_flash_prefill_cp8_ep8_sm90_smoke` | 32K/64K, output 1 |
| D smoke | `smoke:deepseek_v41_flash_decode_dp8_ep8_sm90_smoke` | 32K/64K, output 32 |
| P perf | `perf_test:deepseek_v41_flash_prefill_cp8_ep8_sm90_perf` | 32K/64K, output 1 |
| D perf | `perf_test:deepseek_v41_flash_decode_dp8_ep8_sm90_perf` | 32K/64K, output 512; local batch 1/4/16 |

Standalone decode uses **synthetic KV**. Its length/timing checks do not establish
real PD transfer, answer correctness, EOS behavior or speculative acceptance.
For a timeline, add `--test_env=DSV41_PERF_PROFILE=1` to the perf target. Formal
measurements run before profiling. Determine trace coverage from actual process,
rank and GPU-device metadata. The 2026-11-20 standalone DP8 perf run produced
54 distinct traces across six configurations: two rank0 traces and one trace
for each of ranks1–7 per configuration, all with matching metadata. The six
selected rank0 traces are representative artifacts, not the complete scope.
Preserve all emitted rows and raw trace files from test outputs.

## Actual PD, reuse and semantic checks

The following existing single-node CP4/DP4 full40 targets are retained for
diagnostics. They are **not runnable release gates on the tested 80GiB-class
L20Z development containers**:

```text
//rtp_llm/test/smoke:deepseek_v41_flash_ced_bounded_pd_cp4_dp4_sm90
//rtp_llm/test/smoke:deepseek_v41_flash_dspark_pd_cp4_dp4_sm90
//rtp_llm/test/perf_test:deepseek_v41_sm90_pd_perf
```

These reserve eight GPUs split into four-card P/D roles on one node. On
2026-11-20, both ordinary and DSpark full40 smoke runs failed during decode
initialization with CUDA OOM, before request assertions. The ordinary trace
reached expert weight packing in `mega_sm90.setup_weights`; these are retained
failures, not accuracy results. The same topology perf target must not inherit
an unconditional required status. No sufficient minimum GPU capacity has been
established. Preserve the original targets and OOM logs; do not reduce layers,
substitute the 24-layer golden or relax assertions to obtain a pass.

For these two 80GiB-class L20Z containers, the required full40 semantic and
real-PD release workflow is the **two-node CP8/DP8 fixture and clients** below.
Run both the gamma0/budget0 baseline and gamma5/budget3 speculative configuration,
with actual semantic, reuse and transport receipts. Fixture readiness alone is
not a pass. The matrix names this workflow `two_node_cp8_dp8_full40`.

The full40 semantic client (also called by those correctness targets after
successful startup) checks two short tasks: native
France/Paris continuation (24-token budget; reaching it is allowed) and chat
`17 + 25 = 42` with thinking disabled and natural chat
`finish_reason=stop`. Every answer is checked before three greedy serial repeats
and two concurrent pairs are compared by actual IDs and text. Raw queries and
responses are preserved under `full40_semantics`; generated text is never made a
golden. This helper can target an existing two-node deployment from inside the
container, without launching or changing either service:

```bash
python3 rtp_llm/test/smoke/deepseek_v41_sm90_semantic_checks.py \
  --endpoint http://VERIFIED_PREFILL_IP:PORT --checkpoint /path/to/full40-checkpoint \
  --output-dir /absolute/owned/semantic-run
```

The 24-layer golden is an optional, separate checkpoint fixture. Never run it
with full40 weights or update it to match a changed output. The SM90 DSpark modes
`normal_graph` and `dspark_eager` require that reduced fixture; do not select them
for a full40 checkpoint.

Two-node production topology additionally requires actual CP8/DP8 requests and
transport receipts. Bind both current source/runtime manifests and preserve
correct answers, EOS, native IDs, three deterministic repeats, small concurrency,
actual PD/accepted/verified counters, eight-CP/eight-DP coverage and raw logs.
A source matrix or a standalone role smoke does not fill that coverage gap.

## Two-node CP8/DP8 fixture and clients

Full40 EP4 loading can exceed an 80GiB device during expert packing. Preserve
that failure; do not reduce layers or substitute the 24-layer golden. The
independent eight-GPU roles below support CP8/EP8 prefill and TP1/DP8/EP8 decode.
Build the fixture inside each existing container:

```bash
bazel build --config=cuda13 --repo_env=RTP_LLM_SM90_DEPS=1 \
  //rtp_llm/test/smoke:deepseek_v41_sm90_pd_role
```

Run that built `py_binary` directly in the container. It acquires the existing
canonical global coordinator and eight per-GPU locks itself, using read-only
opens and the same inode/flock semantics as the official wrapper. Do not wrap
it in another `gpu_lock`. Missing lock files fail; none are created or chmodded.
The controller must set `RUNFILES_DIR` and the selected runfiles Torch/NVIDIA
`LD_LIBRARY_PATH` before launch. The binary verifies these paths before its
first SDK import. Do not inherit a build-only wheel download proxy: the fixture
clears HTTP(S)/ALL proxy variables in its own environment and includes both
verified P/D IPs and loopback in NO_PROXY. It normalizes DeepGEMM and keeps JIT/FlashInfer caches in the
explicit owned cache directory without changing `HOME`.

Required CLI fields are `--role`, `--host-ip`, `--port`, `--peer-ip`,
`--peer-port`, `--checkpoint`, `--source-manifest`, `--output-dir`,
`--done-file` and `--run-id`. Use verified container IPv4 addresses and fixed
free ports. Output/cache parent directories must already exist and be owned.
The done file must be absent initially and directly inside this role's output.
Both roles use the same run-id and peer routes. Set gamma0/budget0 for an
ordinary baseline or gamma5/budget3 for bounded/fixed/packed DSpARK; SP verify
flags are applied only to D. TCP mode0 is explicit and is not RDMA validation.
For gamma5, both roles declare `sp_type=dspark` and the same draft checkpoint,
matching the upstream P/D fixture. A non-speculative P cannot provide all cache
entries requested by speculative D. Packed verification flags remain D-only.

Smoke defaults are 1GiB KV, max sequence32768 and D concurrency4. For the
32K/64K throughput grid, provision enough KV (for example8192 or16384MiB),
max sequence69632 and D concurrency16. Graph rows must cover96 target rows for
gamma5; rows above the supported HC64 boundary use the existing fallback.
P context batch remains1 and global concurrency128. Keep budget and cache
configuration identical between comparable timing runs.

`fixture.json` records source/config hashes, physical GPU lock identities,
actual worker routes, owned PID/starttime, argv, logs and ordered Python paths.
`FIXTURE_READY` only establishes readiness, not semantic accuracy. After all
clients finish, atomically create the done JSON
`{"run_id":"THE_SAME_RUN_ID","action":"stop"}`. The fixture stops only its
owned server tree, releases the locks and leaves logs intact; expiry is a
failure. There is no SSH or cross-node process control in the fixture.

Run all clients inside the development container. Reuse the fixture's recorded
`ordered_python_path` for the selected dependency closure, especially the
original checkpoint `AutoTokenizer` used by the reuse contract:

```bash
python3 -m rtp_llm.test.smoke.deepseek_v41_sm90_remote_pd_contract \
  --prefill-endpoint "$P_ENDPOINT" --decode-endpoint "$D_ENDPOINT" \
  --checkpoint "$CHECKPOINT" --output-dir "$NEW_CONTRACT_OUTPUT" --gamma 0
python3 -m rtp_llm.test.perf_test.deepseek_v41_sm90_remote_pd_perf \
  --prefill-endpoint "$P_ENDPOINT" --decode-endpoint "$D_ENDPOINT" \
  --prefill-backend-http-endpoints "$VERIFIED_CP8_BACKEND_HTTP_ENDPOINTS" \
  --output-dir "$NEW_PERF_OUTPUT" --run-id "$RUN_ID" \
  --public-sha "$ACTUAL_FIXTURE_PUBLIC_SHA" --internal-sha "$ACTUAL_FIXTURE_INTERNAL_SHA" \
  --gamma 0 --lengths 32768,65536 --batches 1,4,16
```

Supply the eight verified prefill backend HTTP control endpoints as a comma-separated
list in CP-rank order. The backend HTTP port is `ServerConfig` base+5 (RPC is
base+1); derive ports from the fixture's observed worker layout. The frontend
`/set_log_level` only changes rank0. The performance client therefore requires
an INFO acknowledgement from every P and D backend before timing. Preserve runs
with DEBUG on other ranks as diagnostics, not clean latency baselines.

When `--profile` is enabled, the separate profile batch starts first. Profiling
is armed only after scheduler state shows a real rank0 request actively decoding.
The client saves request IDs, iterations and observation timestamps, including
on failure. Intersect those with actual trace timestamps and rank metadata;
GPU kernels on an EP peer alone can be dummy work and do not prove an active
request on that rank. RuntimeMeta refreshes `phase` live but only fills
`iterate_count` on completion, so the gate requires the same RUNNING request in
two observations instead of a positive in-flight iteration count. Use
`--profile --profile-only` to retry collection independently when the formal
measurements have already completed; retain the original failed collection.

The contract routes requests to all eight D ranks, checks three fresh-prefill
steps and CP8 prefix reuse over at least4096 tokens. It preserves cold/fill/bypass
and repeated-hit comparisons; ordinary decode also checks selected logits and
cumulative scores. Each output directory selects a distinct notebook prefix,
so reusing a running P fixture still exercises a cold fill before cache hits.
Gamma5 uses actual verified-token counters rather than a
made-up denominator. It also preserves the upstream sampling-logprobs contract:
greedy, top-p and full-distribution modes each run twice against the same real
D rank, with unchanged `1e-5` and `places=5` checks. Only greedy repeats must
match; stochastic draft proposals use the global RNG. All six raw requests and
responses are saved. Run the separate semantic client above before perf.
The perf client uses real KV, exact serving-tokenizer lengths and offered
per-rank concurrency; raw artifact writes occur outside the timing boundary.
Add `--profile` for its separate final profile window and retain the flushed
trace. Fixture readiness, semantic answers, reuse/transport and timing each
have separate receipts. Server-side eight-CP transport traces remain a
controller audit; `pd_sep`/routes alone do not prove that trace coverage.

## Result evidence

Record the actual executed argv, paired public/internal SHAs and dirty state,
GPU identity, dependency origins, raw log, final exit signal and Bazel `test.xml`.
An empty signal or entirely skipped target is incomplete validation. For model
and transport tests, keep the raw response IDs and server evidence separately.
Do not infer a 70% acceptance target, production TPOT benefit or distributional
accuracy from short correctness cases. No test command in this document changes
server configuration or runs automatically.
