# K3 MTP validation — 2026-09-07

This implementation is not yet accepted for production. Full 93-layer MTP PD smoke passed at `f0787f94e`; GPU numerical parity,
multimodal requests and EAGLE3/target-only regressions remain incomplete.

Source baseline: `origin/feat/k3_dev` at
`3db35232cc5b004a3b410714ad034795a5790940`.
Development branch: `codex/k3-mtp-vllm028`.
Reference: official vLLM v0.28.0 at
`2cf0a6915ce544dc493a0990f2ea38d81601128a`.

| Check | Result | Scope |
| --- | --- | --- |
| Smoke shell syntax | Passed | `bash -n` |
| Smoke driver tests | 6 passed | Mode/artifact forwarding, confirmed exit handling, transient observation retry and container control paths |
| Checkpoint validator tests | 4 passed | Valid shards, incorrect shape, missing expert, truncated payload |
| Actual draft checkpoint metadata | Passed on 111 | 9 shards, 5400 required tensors, 4 explicit unused AttnRes tensors |
| Python source compilation / diff whitespace | Passed | Syntax only |
| MTP modeling / chunk CPU contracts | 27 passed on 142 at `f167be779` | 8 MTP contracts plus 19 chunk tests; includes absent Decode prefix metadata and Prefill Graph scratch lengths |
| Direct CUDA-build Python import with GPUs hidden | Failed before tests | Existing compute_ops fallback cannot provide FusedRope symbols when CUDA is unavailable; CPU harness supplies fail-fast constructor sentinels, never claims kernel validation |
| Current config and checkpoint weight-name mapping | Passed | One runtime layer, explicit MoE local0, source93, 5400 required names, no AttnRes |
| C++ independent pool configuration | 2 passed | EAGLE3 and K3 MTP; candidate counts 1 and 3 both produce one MTP sub-config |
| Current extensions / executor build | Passed | Independent CUDA13/SM10x builds; 106/142/110/112/113 at `41930af44` |
| GPU MtpExecutor fixtures | 23 passed on 113 | Official Bazel test runner, CUDA device 0, benchmark excluded; chunk lookahead, media token restoration, verify/update and cache state operations |
| Actual GPU loading and Decode Graph initialization | Passed four-layer startup on 142 at `f167be779` | Positive FastSafetensors logs for target/draft and initialized MTP Graph; does not prove numerical parity |
| Stage tensors, recursive h/z numerical parity, physical pool addresses | Pending | Runtime stage comparison and address evidence still required |
| RDMA readiness tests | 2 passed | All eight rank initialization records required; replay confirms the previous first request preceded seven listeners |
| Four-layer TP8EP8 PD flow | Passed on 106/142 at `41930af44` | Input 4,987 tokens, chunk size 4,096, PD enabled, 256 output tokens; P/D/controller exit 0 |
| Full 93-layer MTP PD all | Passed at `f0787f94e` | 31 cases, both roles and controller exit 0; see run evidence below |
| EAGLE3 and target-only GPU regression | Pending | Required before broader acceptance |

The actual draft is the separate
`Kimi-K3-MTP-HASS-iter_0001281-vllm-draft-only` checkpoint on the local data disk.
Header verification is read-only and does not read all tensor payloads or prove
fastsafetensors loading, numerical fidelity or effective target/head binding.

Build host: L20-dev-111 (B300/SM10x), `lhc_GPU`, user `luohaocheng.lhc`.
Source: `/data1/luohaocheng.lhc/code_repo/k3-mtp-vllm028` (ext4).
Output: `/data1/luohaocheng.lhc/.cache/bazel-k3-mtp/38c41e724692707333b194f1344581f6` (ext4).
All build commands use `--config=cuda13 --config=sm10x`; CPU-only test attempts
set `CUDA_VISIBLE_DEVICES=`. The original working checkout is unchanged.
Internal build metadata was exported separately from the same branch at
`f87b8e377438df0fc2c4ce83ba532f9d65f2a3ee` after the existing 111 metadata proved
too old for the public source's `tokenspeed_mla_test_deps` symbol. Both
`stub_source` and `internal_source` now resolve to this isolated export.
The initial missing cuLA dependency was an old `internal_source` path, not
a missing entry in the latest lock. The retry uses unmodified hashes and
clears stale proxy variables only for its own process/repository environment.
The offline xgrammar source is `3842647890df7c8133fba6bc0e3d11fc9730e0bd`,
with dlpack `bbd2f4d32427e548797929af08cfe2a9cbb3cf12`.

## GPU executor and PD preparation

On L20-dev-113, `bazelisk test --config=cuda13 --config=sm10x`
ran all 23 non-benchmark `MtpExecutorTest` cases successfully. The test used
`lhc_GPU` as `luohaocheng.lhc`, the host's own build, and
`--test_env=CUDA_VISIBLE_DEVICES=0`. Immediately before launch, all eight GPUs
had no compute processes, 0% utilization and approximately 268 GiB free.
The earlier direct-binary attempt on 111 aborted before its first test because
Bazel test environment variables were missing; it is not counted as a pass.
The successful log is `k3-mtp-pd-runs/preflight-113/k3-mtp-executor-test.log`
in the controller workspace. These fixtures exercise executor operations and
CUDA cache kernels, not a loaded K3 model or vLLM numerical parity.

Both L20D-dev-106 and L20D-dev-142 independently completed all 37,108 build
actions for source `743edf08c`. Source and build outputs reside below
`/ssd/1/luohaocheng.lhc/k3-mtp-20260907`; no compiled outputs were copied between
hosts. The full target at `/ssd/5/kimi-k3`, four-layer target and separate
MTP checkpoint have matching configuration, index and every shard SHA256 on
106/142 and 110/113. All three checkpoint preflights passed inside each B-group
runtime container, using explicit local roots `/ssd/1` and `/ssd/5`.
The smoke views correct index `total_size` metadata without changing payloads.

Host selection remains dynamic. A machine's completed build does not reserve
its GPUs. 142 became occupied before its executor launch, so that launch was
cancelled and testing moved to eligible 113. After the user explicitly authorized
cleanup of zero-utilization residual processes, identified residual services were
stopped with a fresh all-GPU idle check; containers remained running.
Full PD acceptance still requires an eligible pair in either 110–115 or
106/120/142/144/145/154, never a pair crossing these groups. The prepared launch
helpers retain each endpoint's local weights and build. Four-layer `flow` passed;
93-layer `all` passed on 106/142 in run `k3-mtp-all-20260907-f0787f9-b3`.
The first (`743edf0-b1`) failed during MTP Graph initialization because plain
Decode supplies no `prefix_lengths`. Commit `f167be779` fixes optional metadata
and distinguishes Prefill Graph scratch sequence lengths; 27 CPU contracts pass.
The second (`f167be7-b2`) loaded both models, initialized Decode Graphs and
completed an actual PD request, but flow correctly rejected input length 4,987
against the configured chunk threshold 65,536. Flow now uses 4,096 for
the four-layer fixture; full `all` retains 65,536.
The third run failed because Prefill rank 0 reported HTTP readiness before
the other seven RDMA listeners were ready. The later buffer timeout followed
the initial connection failure and request expiry. Commit `41930af44` waits
for all eight RDMA initialization records before sending smoke requests.

Run `k3-mtp-flow-20260907-41930af-b4` passed with input length 4,987,
chunk size 4,096, `pd_sep=true`, and 256 output tokens. Both remote roles
and the controller exited 0. Its `accuracy.json`, role logs, and separate
Prefill/Decode evidence archives are retained in `k3-mtp-pd-runs`.
Runtime logs show target MLA pool 0, target KDA pool 1, and draft MLA pool 2.
The truncated target produced zero accepted draft tokens; this flow proves
chunked transport and execution, not full-model accuracy or useful acceptance.

The controller now reads detached control files inside the runtime container.
Pouch isolates `/tmp` from the host, so host-side polling previously missed
completed role status. Confirmed role exit also ends the readiness wait; an
SSH observation failure alone does not.

Full run `k3-mtp-all-20260907-41930af-b1` passed 13 semantic cases, including
cache misses, full and partial prefix hits, and batch requests. Individual
cases accepted 25–91 draft tokens. The overall run failed a mixed-request
cache-miss assertion after an intrusive debugger attachment paused workers:
the route log records a 125,239 ms keepalive watchdog timeout and retries,
so the retried prompt legitimately reused cache. This run is not a full pass.
The case runner now uses unbuffered output (`c79812912`) so progress can be
observed without pausing the service. No cache assertion was weakened.

Run `k3-mtp-all-20260907-c798129-a2` failed during Decode weight loading on
112. Another service appeared after the idle preflight and occupied about
99 GiB per GPU; the task's process then exhausted the remaining memory.
This does not establish an OOM under exclusive GPU use.

Run `k3-mtp-all-20260907-f0787f9-b3` passed on 106 (Prefill) and
142 (Decode), following a fresh idle preflight with about 268.6 GiB free per
GPU. Optional `SMOKE_PARALLEL_START=1` starts both roles loading concurrently;
role health, RDMA and result-listener readiness gates remain in effect.
The default controller startup order is unchanged. Both roles and the
controller exited 0; `accuracy.json` reports `passed=true`, 31 cases and
15 stages, including six concurrent stages. Every case has `pd_sep=true`.
The MTP chunk case used 90,132 input tokens with a 65,536-token chunk
threshold, generated 90 tokens in 27 iterations, and therefore accepted
63 draft tokens. Long-input single and batch cache-hit cases reused
90,112 tokens. All 31 cases accepted draft tokens (25–90 each).
Both role archives contain eight RDMA-ready records, startup configuration,
service logs, engine logs and successful role summaries. Artifacts are in
`k3-mtp-pd-runs/k3-mtp-all-20260907-f0787f9-b3` in the controller workspace.

This completes the full-model two-host smoke gate. Remaining broader plan
validation includes EAGLE3/target-only regression, multimodal requests and pinned
vLLM/RTP stage tensors, physical cache addresses and memory consumption.
Acceptance counts alone do not establish numerical correctness.

## CPU contract reproduction

With the current built extensions and locked Python dependencies on PYTHONPATH:

```bash
CUDA_VISIBLE_DEVICES= python3 example/k3/run_kimi_k3_mtp_cpu_contract_tests.py
```

This runner substitutes only the three unavailable FusedRope constructor names
with sentinels that raise if constructed. Attention/MoE in equation tests are
explicit CPU test doubles. It does not spoof CUDA availability, execute GPU
kernels, compare against a running vLLM model, or validate collective ordering.
The binding test checks that Python model creation follows target embedding/head
binding and that draft norm ownership is preserved. Mode/CP rejection, media
placeholder restoration and chunk feature slicing are covered.

The official reference's multimodal capability check changed one assumption in
the initial plan: K3 MTP does not receive target visual embeddings in vLLM 0.28.0.
RTP draft consumes media token embeddings and target hidden; the target continues
to use the visual embeddings. See the contract table for exact reference calls.

## Hardcoding review — 2026-09-08

Review covered all 29 files changed from the pinned `origin/feat/k3_dev` base,
plus the PD launcher used by the smoke. Four issues were corrected:

- MTP config, validation and weight names required source layer 93. They now
  derive the source layer from the draft's `num_hidden_layers`, matching
  vLLM's `mtp_start_layer_idx`; runtime local0 and the combined cache layer
  remain separate indices.
- Validation required exactly nine shards and a quantization group named
  `group_0`. Shards now follow the index and quantization groups are checked
  by their format. Missing experts, incompatible shapes/dtypes and truncated
  payloads still fail.
- RDMA readiness required eight ranks. The existing launcher also forced TP,
  EP, world size, local world size and visible devices to eight. These now
  follow `KIMI_K3_TP_SIZE` / `KIMI_K3_EP_SIZE`. The smoke forwards them to both
  roles, sizes Prefill MegaMoE token capacity from chunk size / TP and disables
  optional shared-expert weight sharding for odd TP sizes.
- MTP smoke inherited EAGLE3's 4/93-layer aux profile check and overwrote
  proposal count with three. Aux checks now apply only to EAGLE3, explicit
  EAGLE3 aux IDs are forwarded, and candidate count follows configuration.
  Target checkpoint path is required explicitly; the inherited `/ssd/2`
  default was removed to prevent selecting an unintended checkpoint.

Runtime H, the 2H fusion input, MLA cache width, head partitioning and expert
partitioning follow model and parallelism configuration. The H-versus-3H
feature distinction is part of MTP/EAGLE3 semantics. An independent MTP cache
pool and one reused module per proposal sequence are intentional. No cached
answer, synthetic acceptance count or suppression of a model error was added
to pass the full smoke.

Remaining implementation boundaries are explicit: one nextn layer, NoPE full
MLA, SiTU/LatentMoE, BF16 compute with checkpoint-native group-32 MXFP4, matching
TP/EP/world groups, and no Prefill CP. The existing whole-chunk path also rejects
Prefill Graph. Removing these checks would not implement the missing modes.
The smoke retains its existing B300 and memory/admission profile defaults;
those values do not define MTP tensor shapes or its forward equations.

Validation of the review fixes: six checkpoint tests cover source layers
4/47/93, shard counts 1/3/9, missing experts, wrong source/shape and truncation.
The Linux launcher dry-run covers both roles at TP1/2/4/8, including world and
local-world sizes. All 28 CPU contracts passed on 142; the new manifest test
covers source layers 4/47/93 and every rank at TP1/2/4/8. Eight driver tests and
two RDMA readiness tests passed. These are configuration, manifest and CPU
contract tests, not non-TP8 GPU evidence. The earlier full GPU pass is pinned
to `f0787f94e`, before these review fixes.

The CPU harness does install fail-fast sentinels for three unavailable GPU
constructors when GPUs are hidden. This is confined to the test entry point;
it does not alter production imports or implement a fallback kernel. Numerical
CPU equation tests use explicit attention/MoE doubles. Neither test technique
is evidence of vLLM GPU numerical parity.


## FP8 rebase and full PD smoke — 2026-09-08

Rebased `codex/k3-mtp-vllm028` onto the fetched
`origin/feat/k3_dev@b0ee8c10a`. Runtime commit: `0986ea1fc`.
The old branch was retained as `codex/k3-mtp-before-rebase-20260908`.
The rebase preserved upstream FP8 loading/MLA changes, the MTP latent-norm
epsilon, configurable topology and source-layer mapping, and both sets of
smoke-driver tests.

The latest upstream smoke already defaults to attention weight FP8, dense MLA
FP8, FP8 collective GEMM, unit Q/KV scales, a 4 GiB prefix-expansion budget,
Prefill TP8EP8, Decode TP8EP8 and Decode CUDA Graph (capture batches 1–4).
These defaults were retained. EAGLE3 remains the default speculative mode;
MTP is selected explicitly using `SP_TYPE=mtp`, `SP_MODEL_TYPE=kimi_k3_mtp`
and the independent local `SP_CHECKPOINT_PATH`. The launcher's obsolete
“cache precision: bf16” message was corrected. BF16 activation/head output and
native MXFP4 experts remain part of the FP8 attention configuration.

Both hosts built the same commit independently inside `lhc_GPU`, as
`luohaocheng.lhc`, with `--config=cuda13 --config=sm10x --jobs=24`.
Source and Bazel output were on local ext4 SSD1. Both builds completed
successfully with 2,381 actions. No binaries or compiler caches were transferred.
Runtime used the existing same-image container
`lhc_GPU_k3_chunkwise_smoke_20260821`.

Target: `/ssd/5/kimi-k3`, 96 indexed shards, approximately 1.42 TiB.
MTP: `/ssd/1/luohaocheng.lhc/k3-mtp-20260907/models/Kimi-K3-MTP-HASS-iter_0001281-smoke`,
nine shards, approximately 20 GiB. Both endpoints passed the local-filesystem,
index/header and explicit FastSafetensors preflight. Each rank0 startup log
contains two positive `finally choose load method: fastsafetensors` records,
for target and draft. The post-launch guard found no alternate loader or
weight-loader fallback. New source, build outputs and logs stayed on SSD1;
no writes were made to SSD2, SSD3 or `/3fs-data/3fs/mtp_test`.

| Check | Result |
| --- | --- |
| Python MTP/FP8/chunk contracts | 37 passed on each host |
| C++ FP8 cache byte format and independent MTP pool | 2 passed on each host |
| Driver / RDMA / checkpoint tests | 9 / 2 / 6 passed |
| Linux launcher topology dry-run | 3 passed, including TP1/2/4/8 on both roles |
| Four-layer FP8 PD flow | Passed: input 4,991, chunk 4,096, output 256, PD true |
| Full 93-layer FP8 PD all | 31/31 passed across 15 stages; 6 concurrent stages, batch size 4 |
| Full-run PD and MTP activity | Every case has PD true; recorded accepted tokens range 25–93 |
| MTP chunk case | Input 90,135, output 90, 27 iterations, 63 accepted draft tokens |
| Long-prefix reuse | 90,112 tokens for single and concurrent hits |
| Remote role exit status | Prefill 0, Decode 0 |

The FP8 tests cover target/MTP enablement with EAGLE3 excluded, preserving
native MoE quantization, and source-layer remapping through the online FP8
wrapper's actual source weights at TP1/2/4/8. The CPU test harness uses the
already documented fail-fast FusedRope sentinels. Initial test invocations
failed on a missing local helper and on invoking the C++ wrapper without its
binary argument; the corrected final test combinations exited 0 on both hosts.
No production GPU backend or test assertion was replaced.

Runtime logs contain 332 rank0 FP8 weight conversion records per role,
including MTP local0 after draft loading. Decode logs contain 64 successful
capture records: every rank 0–7 captures batches 1–4 for target and MTP.
Separate model-scoped FMHA workspaces are recorded for both models.
The combined cache configuration has one MTP sub-config and three independent
groups:

| Pool | Layers | Prefill blocks | Decode blocks | Physical state |
| --- | --- | --- | --- | --- |
| Target FULL MLA | 24 | 382 | 640 | Ordinary E4M3, 576 bytes/token |
| Target KDA | 69 | 382 | 112 | FP32 SSM + BF16 convolution |
| MTP FULL MLA | 1 | 382 | 640 | Ordinary E4M3, 576 bytes/token |

Both MLA specs report `fp8_plain=1`, 4,096 tokens per block and 2,359,296
bytes per layer/block. KDA reports 814,080 bytes per layer/block. Decode's
logged free memory immediately after graph capture was 6,091–7,429 MiB across
ranks; this is one initialization observation, not a peak-memory measurement.

Full run: `k3-mtp-fp8-all-0986ea1-20260908-a1`.
Flow run: `k3-mtp-fp8-flow-0986ea1-20260908-a1`.
Both used the normal role script and `SMOKE_PARALLEL_START=1`. While the full
models were initializing, WebTerminal auth expired. The local controller
exited 1 after repeated status-poll failures. Its failure log is retained.
One browser-backed login restored auth and the existing personal-account SSH
masters. A read-only reattachment read the original role status files; both
were 0 and the reattachment exited 0. Remote services were not restarted and
smoke requests were not replayed. This distinguishes the control-connection
failure from the successful model/PD acceptance run.

Controller artifacts under `k3-mtp-pd-runs/<run>/` include `accuracy.json`,
original role logs, `reattached-status.json`, filtered role environment JSON
and verified role evidence archives containing startup, cache, graph and RDMA
logs. Build/preflight/final CPU logs are under
`k3-mtp-pd-runs/fp8-preflight-20260908/{106,142}/`.
Both hosts had no GPU compute processes after role cleanup.

This result covers the requested FP8 MTP smoke. Stage-by-stage vLLM numerical
parity, non-TP8 GPU execution, multimodal and separate EAGLE3/target-only GPU
regressions remain distinct items in the broader validation plan above.
