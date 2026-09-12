# K3 MTP native attention precision validation

Status: CPU contracts, C++ cache/executor tests, four-layer PD flow and full
93-layer PD smoke passed. The fixed 32-request HumanEval subset passed with
5,614 accepted / 7,581 proposed draft tokens (74.05% raw verify acceptance).

Base: `origin/feat/k3_dev@9a5acff9a9ba53932e9670eb5964b9791024e3ee`, fetched
2026-09-11. Work branch: `codex/k3-mtp-native-precision-20260911`.

## Contract

Only `kimi_k3` consumes the two K3 FP8 switches. MTP uses its checkpoint BF16
compute dtype, unquantized attention projections, BF16 MLA and BASE KV cache.
Its native MXFP4 experts and MegaMoE FP8 activation compute are unchanged.
No speculative modeling, recurrent hidden, token shift or acceptance logic changes.

The weight manifest rejects inconsistent runtime precision before constructing
any online FP8 weights. Initialization ignores process-wide cache flags for
MTP without mutating shared configuration. Both PD roles log final precision.

## Completed CPU checks

At `1fcbf51`, the CPU contract runner completed 52 tests: 51 passed and one
CUDA-only pinned-memory test skipped. It used the revised Python source and
existing same-host native bindings on 110, with `CUDA_VISIBLE_DEVICES` empty.
This does not validate the revised C++ implementation. No GPU kernel ran.

The common precision initializer was real, including the global cache flag
path and checkpoint configuration lookup. Tests cover all four K3 switch
combinations, all FP8/INT8 cache-flag combinations, all six model creation
orders, repeated initialization, stale QuantAlgo state and rejection of runtime
quantization. The native manifest covers source layers 4/47/93 and TP1/2/4/8;
existing target FP8/Projection-KTP split tests also passed. Dense attention
routing called the FP8 quantizer three times in the FP8 branch and zero times
in the BF16 branch, using mocked attention kernels only.

The Linux launch/PD-driver suite passed 34 tests at `13cafa8` and all 35 tests
at `b3719bc` after the explicit owner-count validation was added. That revision
selected TP/EP Decode for MTP to match the earlier plan. The current acceptance
topology is Prefill TP8EP8 and Decode DP8KTP8EP8. Baseline `9a5acff` already
supports MTP target verification with KTP; its old smoke documentation saying
otherwise was stale. The MTP-only topology fallback is removed. Both draft
modes retain KTP1 and must produce KTP graph replay and PD fan-in evidence.
The explicit owner-count tests also retain support for separate single-owner
experiments; those do not count as DP8 acceptance.

## Checkpoint provenance

Both 106 and 142 passed the local-storage/header guard for `/ssd/5/kimi-k3`,
the existing four-layer checkpoint, and the MTP `iter_0001281-smoke` directory
under `/ssd/1/luohaocheng.lhc/k3-mtp-20260907/models`. The MTP smoke directory
shares all nine shard files with the original `vllm-draft-only` directory
(`os.path.samefile` is true for each file). Config and weight maps are equal.
Only index `metadata.total_size` differs: the original counted file headers
(21,470,681,944 bytes); the corrected index records tensor payload
(21,469,868,544 bytes). The original index fails the guard; it was not changed.

Builds run inside `lhc_GPU` as `luohaocheng.lhc` with CUDA13/SM10x. Model tests
will use the existing same-image `lhc_GPU_k3_chunkwise_smoke_20260821` container,
which exposes SSD5 read-only; the base build container does not mount SSD5.
Source and builds use SSD1; current service logs and JIT caches use container `/tmp`. No files were written to SSD2/3 or the
protected MTP test storage.

## Cache audit

`createSpConfig` creates target and draft specs separately, adds both to the
capacity calculation, and appends the draft FULL pool. `createConfigForGroup`
uses that group's byte stride and spec. P2P layer conversion passes allocator
`BlockInfo.size_bytes` to transport; no target dtype conversion is needed.
The test-only `setKVBlockValue` helper incorrectly used `cache_specs[0]` for
all layers; it now selects the layer's group. The mixed-pool allocation test
checks the BF16 draft payload by writing and reading distinct K/V bytes.

## Completed native build and C++ checks

Both 106 and 142 independently built all four targets successfully at `1fcbf51`
inside `lhc_GPU`, using CUDA13/SM10x and 24 build jobs. The subsequent commits
only change smoke launchers and their Python tests; native model code is unchanged.
Build durations were 2448.821 seconds and 2539.650 seconds, respectively.

On 106, the cache-spec test (1 case), cache geometry suite (3 cases), and MTP
executor suite (26 correctness cases) passed on CUDA. The new mixed-pool case
executes TP1/2/4/8 and candidate counts 1/3, with real allocations and byte
readback. The executor microbenchmark was excluded; no performance result is
reported. Direct test launch supplies the Bazel TEST_SRCDIR/TEST_WORKSPACE/
TEST_BINARY variables and the same-host runfile library paths. Its first executor
attempt exited before tests because those variables were absent; the corrected
run passed all 26 cases.

142 SSD1 filled completely after the build. Three task-owned test executables
were copied to container `/tmp`, SHA256-verified and removed from the build
output to free about 3.4 GiB. The server executable and weights were retained. The task-owned output root is
about 49 GiB; the filesystem continued growing after our build finished. No
external files/processes were removed. Test logs/temp files use container `/tmp`,
whose backing local filesystem had 2.6 TiB free. Full service launch must recheck
storage, JIT-cache destinations and GPU ownership.

## Model startup precheck and current acceptance target

A four-layer Decode TP8EP8 startup at `9d9dbbb` reached READY on 142.
All eight rank logs showed target FP8 projection/MLA/cache and MTP native BF16
attention with BASE cache. FastSafetensors log verification passed, standalone
MTP checkpoint validation reported nine shards and 5,400 required tensors, and
target embedding/head binding completed. Target and draft CUDA Graph setup
completed. This was a loading/initialization check only: no PD requests ran,
and it does not validate the requested DP8KTP8 topology. The task-owned service
was stopped when the acceptance topology changed.

The current smoke revision is `b3719bc`: Prefill TP8EP8, Decode DP8KTP8EP8,
Decode CUDA Graph, target FP8 and MTP native BF16 attention/cache. Both 106 and
142 source trees have this revision. Their native build comes from the same
precision implementation; later tracked changes are launcher/test/documentation
changes. The completed full-model PD and HumanEval results are recorded below.

A task-only HumanEval audit patch on 106 observes the already-ready CPU
`accept_len`, records `accept_len - 1` accepted draft tokens and the configured
candidate count, and tags each event with the request/trace ID. Patch SHA256:
`db327594ecc848922f027b22280e83b0038bd0890fe0c959b719e856b8d2862b`.
It adds no attention operation, sampling change or additional GPU synchronization.
It is not part of the committed precision fix. The diagnostic server rebuild
passed inside 106's `lhc_GPU` (exit 0, 67.980 seconds).
Raw sampler acceptance and emitted-token counts must be reported separately
at EOS/max-token boundaries.

Only system device-health probes are authorized to coexist with correctness
runs. Repeated live probes found unrelated `sol-execbench`/Ray jobs on 106,
active inference on 110/112/113/115, and no Pouch permission on 144/145. These
processes were not stopped. 111 and 142 are individually available but belong
to different data-plane groups. Enabling the shared-accuracy selector does not
authorize coexistence with unrelated model or operator jobs.

The next available same-cluster pair was 142/145. The 145 task checkout is
`/ssd/5/luohaocheng.lhc/code_repo/K3-mtp-native-precision-20260911` at `69abc43`,
with the same task-only audit patch. Its independent CUDA13/SM10x build uses
24 jobs inside `lhc_GPU`; the build passed in 1228.579 seconds (exit 0).
Runtime uses `lhc_GPU_k3_rdma_20260908_1`; both runtime
containers expose eight active `mlx5_bond_0` through `mlx5_bond_7` interfaces.
Existing personal-account login sessions on 145 had different supplementary
groups. A user-specific socket ACL now permits `luohaocheng.lhc` to access
Pouch without relying on old session group membership. Fresh access was verified.

145's full target, four-layer target and MTP header/storage guards passed.
The full target index SHA256 matches 142:
`a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`.
Four-layer target and MTP configs match across the two hosts. Its native MTP
test directory under `/ssd/5/luohaocheng.lhc/models` also fixes only the index's
payload size; all nine shards share inodes with `/ssd/5/kimi-k3-mtp`.
These checks establish launch prerequisites, not a completed PD run.

Four-layer flow `native-ktp-flow-20260911-c` passed on 142 Prefill / 145 Decode;
both detached role statuses are 0. It used the requested TP8EP8 / DP8KTP8EP8
topology, CUDA Graph, target FP8 and native BF16 MTP attention/cache. Both
FastSafetensors log guards passed. The request contained 4,982 input tokens
across two 4,096-token chunks and exercised one Decode owner. KTP graph replay,
PD fan-in and graph buckets 1/2/4/8 were present. This is connectivity and
execution coverage with a truncated target, not full-model accuracy or an
acceptance-rate benchmark. Its actual audit counts were 0 accepted / 768
proposed tokens; the response-length approximation incorrectly yields -1.

## Full-model PD and HumanEval results, 2026-09-11

Full 93-layer `all` run `native-ktp-full-20260911-a` passed on Prefill 142 and
Decode 145. Both roles reported PASS. The topology was Prefill TP8EP8 and
Decode TP1DP8KTP8EP8, with draft TP1DP8KTP1EP8, three candidates, Decode CUDA
Graph and both target FP8 switches enabled. The checkpoint loader was explicitly
FastSafetensors; both post-startup guards passed. No capacity or sequence-length
setting was reduced to obtain this result.

The suite passed all 52 cases: 17 cache hits, 35 misses, eight concurrent stages,
two MTP-specific cases and one multimodal case. It covered graph bucket 8,
uneven owner batches, partial-prefix reuse, whole-chunk reuse and long prefixes.
The maximum input was 100,044 tokens; the multimodal MTP input was 95,627 tokens.
Whole-chunk hits reused 90,112 tokens. The service exposed all eight Decode
owners and all eight ranks participated in KTP/EP; actual requests in this
suite reached owners 0–3. This does not claim request traffic to every owner.

All eight rank logs on each host show target `fp8_per_block`, MLA FP8 and FP8
cache, alongside MTP BF16, attention quantization `none`, MLA FP8 disabled and
cache `BASE`. Native MXFP4 experts and MegaMoE FP8 activation compute remain.
The cache specification gives each target MLA layer 2,359,296 bytes per
4,096-token block and the independent MTP MLA layer 4,718,592 bytes. Decode
allocated 107 MTP blocks with view `[107, 4096, 576]`; the draft payload is
therefore 504,889,344 bytes per rank. BF16 doubles the draft payload compared
with FP8 at the same block count. Target LINEAR/KDA storage is unchanged.

HumanEval ran on the retained service pair after smoke, with dataset revision
`6d43fb980f9fee3c892a914eda09951f772ad10d`, shuffle seed 20260908,
one bootstrap request, 10 warmup requests and 32 measured requests.
Concurrency was 1, maximum output 256, temperature 0 and top-k 1. All measured
requests passed PD and zero-prefix-reuse assertions; inputs ranged from 145
to 379 tokens. Warmup and smoke requests are excluded from the acceptance
counts. This evaluates draft acceptance on code prompts, not HumanEval pass@1.

The task-only Decode audit recorded 2,527 verify events: **5,614 accepted draft
tokens / 7,581 proposed draft tokens = 74.0536%**. Each trace maps to exactly
one Decode stream; there are no duplicate events or missing measured traces.
For every request, reported iterations equal verify events plus the initial
Prefill step. Sequence advancement agrees with raw acceptance until terminal
trimming; each final in-flight event starts at the emitted-output boundary.
The audit counts raw verification before EOS/max-token trimming, including
that final in-flight event. The 8,025 emitted output tokens and 8,141 raw verify
tokens are separate quantities and must not be substituted in the acceptance
formula. No before/after acceptance or throughput improvement is claimed.

Two-second `nvidia-smi` samples observed maximum used memory of 269,984 MiB
(263.66 GiB) on Prefill and 267,869 MiB (261.59 GiB) on Decode. Sampling started
after loading began, so these are sampled process/device peaks, not exact
allocator or complete-startup peaks. No CUDA OOM, abnormal exit or restart was
found during smoke or HumanEval. The controlled shutdown after completion is
not a test failure.

Local evidence is retained under `artifacts/k3-native-precision-20260911/`
(untracked): final per-host log archives, `full-accuracy.json`, HumanEval request
and acceptance reports, canonical Decode audit log, rank precision summaries,
cache evidence, sampled-memory data, launch scripts and the diagnostic patch.
The audit patch is not part of the production precision fix.

The original production OOM request was unavailable. The 100K smoke is not an
exact replay of that incident. Separate 128K/OOM replay, EAGLE3 and target-only
GPU regressions, a same-checkpoint numerical golden comparison, and a matched
old-FP8 memory/acceptance baseline remain outside these completed run results.

## Validation to record

- Real common precision initialization: switch/cache matrix, shared objects,
  initialization permutations, repeated initialization, invalid dtype/quantization.
- Native MTP manifests across source layers and TP ranks; target FP8 regression;
  binding and recurrent-state contracts; BF16 dense attention dispatch.
- Actual C++ allocation: target FP8 FULL, native LINEAR, MTP BF16 FULL;
  independent pointers, layer mapping, per-layer bytes, one recurrent module.
- Four-layer flow and full model PD: Prefill TP8EP8, Decode DP8KTP8EP8 with
  Decode CUDA Graph; EAGLE3 and target-only regression.
- Original OOM request or documented geometry substitute; 32K/64K/128K requests;
  measured memory and fixed HumanEval subset acceptance counts.

No OOM or acceptance claim follows from the static/config tests. BF16 draft
KV doubles its payload relative to ordinary FP8; expanded BF16 workspace may
still dominate peak memory. Do not reduce request geometry to claim a pass.

## Deployment

On 2026-09-12 the controller and direct role script were changed to default
to MTP. Controller commands now explicitly carry `SP_TYPE=mtp` and
`SP_MODEL_TYPE=kimi_k3_mtp` on both endpoints even when the caller supplies
neither variable. Startup examples specify both variables and MTP checkpoints.
The earlier full-model run already supplied these exact MTP values; its two
archived `service.env` files were rechecked. This default-only change was
validated with 27 controller/request-suite unit tests, shell syntax checking
and a two-role dry run; no new GPU run is claimed.

The work branch history is grouped into precision isolation, smoke configuration
and a separate test commit. Historical hashes above identify the actual tested
revisions before that history rewrite; archived logs preserve those identities.

The minimum image revision will be the tested precision-isolation commit.
Upgrade or roll back both PD roles together and recreate caches. FP8 draft
cache from old instances is incompatible. Whale deployment is outside this task.
