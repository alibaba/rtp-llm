# Framework integration checkpoint

This checkpoint builds on the [shared-storage component checkpoint](CHECKPOINT_VALIDATION.md)
from public main `492ecf45b3ef55b22fd4882622abfdb2e2f664f2`.
It adds an experimental file bootstrap and model loading hook. It contains no
service layer, platform injection, Owner RPC, failover or live generation switch.

## Runtime path and trust boundary

```mermaid
sequenceDiagram
  participant L as Trusted launcher
  participant O as Test Owner, host A/container A
  participant C as RTP consumer, host B/container B
  participant G as Consumer GPU
  O->>O: Inspect safetensors; create and load HOST_NUMA/FABRIC allocations
  O->>L: Private bundle + manifest + opaque capabilities
  L->>C: Authenticated transfer into private files
  C->>C: Actual tokenizer/hash ABI + layout checks
  C->>G: Import/map and verify selected rows
  G-->>C: GPU samples match checkpoint-derived digests
  C->>C: Construct external Engram layers transactionally
  G->>O: Direct memory reads through mapped fabric path
  Note over G,O: No lookup RPC or Owner CPU request handler
```

`RTP_ENGRAM_EXTERNAL_BUNDLE` is an absolute path read once during initialization.
When absent, legacy loading remains the default. The trusted launcher must choose
the correct bundle, authenticate its transfer, protect its files and retain the
Owner until every consumer process has exited. Bundle files must be regular,
current-UID files with no group/other access; final-component symlinks are refused.
These checks do not replace authenticated distribution or a platform trust model.

The bundle pins a manifest digest, generation, dataset, model revision, hash ABI
and required read samples. The consumer recomputes actual tokenizer/hash
semantics and validates the complete layout before mapping. **Dataset and model
revision are supplied by the trusted bundle; they are not yet independently
pinned by local model configuration.** A launcher selecting the wrong bundle with
compatible geometry/hash semantics is outside the implemented protection.

`engram_model_attach.py` attaches once, constructs all selected model layers in a
temporary list and publishes only after complete success. Partial failure releases
views before closing storage; incomplete cleanup retains resources and fails.
There is no silent private-table fallback or in-process retry after failed init.

The model integration is **eager only**. External mode rejects configured engine
Graph use, Graph preparation and capture detected at forward entry. Successful
model mappings are retained for worker-process lifetime. There is no live detach,
model unload or engine-wide Graph cleanup contract. The older isolated guarded
Graph harness does not establish Graph support for this model hook.

## What was exercised on two hosts/containers

The current fixture revision is `engram-framework-fixture-main-492ecf45-v2`.
Earlier v1 evidence is retained with its narrower coverage described below.

| Check | Result | Scope |
| --- | --- | --- |
| Real v2 tokenizer and safetensors fixture | Passed | Actual tokenizer reload and tensor dtype/shape checks; two Engram layers |
| Bootstrap UT | 29 passed on each host, zero skipped | File/identity/hash and cleanup contracts; GPU mapping mocked where appropriate |
| Model attachment UT | 19 passed on each host, zero skipped | Transactional initialization, legacy behavior and eager-only guards |
| Model-hook harness CPU UT | 16 passed on each host, zero skipped | Bounded native preflight, fixture checks and fixed comparison bounds; no model forward |
| v2 fixture CPU UT | 10 passed on each host, zero skipped | Rolling decode history, masks, selected rows and all-row uniqueness |
| v2 cross-host bootstrap, hash and lookup | 12 layer/case checks passed exactly | Real mapped GPU reads and RTP hash/lookup; excludes WKV/model hook |
| Bad sample digest fault, v1 fixture | Rejected with `GPU_READ_SAMPLE_MISMATCH` | Real remote GPU reads fail verification; consumer native counters return to zero |
| Main native libraries | Build and required imports passed | Genuine config and compute extensions; no projection shim |
| Actual v2 model hook and layer forward | 12 cases passed locally and 12 across hosts/containers | Real loading/preparation hooks, original quantization/DeepGEMM and generic gate; no complete LLM |
| Bazel CPU targets | 19 + 29 passed on the Owner host, zero skipped | Original main wrapper, `GPU_COUNT=0`, `--nocache_test_results` |
| Full model/engine/HTTP | **Not tested** | No attention/MoE/scheduler or serving acceptance |

The deterministic fixture has a real 32-token tokenizer, two layers with
4,097 and 514 embedding rows, 64 columns and group-32 scales. The owner checkpoint
contains 12 tensors. The consumer checkpoint contains only 8 local WKV/q/k
tensors plus tokenizer/configuration. Six v2 input cases cover decode steps
0/1/7 with 0/1/3 valid history tokens, mixed masked/boundary input, 33-token
prefill and an empty batch. Each singleton decode window comes from the same
rolling sequence. A layer/row sign signature makes all 4,611 table rows distinct
both as stored weight/scale bytes and after dequantization.

The fixture has **HC=2 and hidden dimension 128**. Its gated residual uses the
generic/reference computation in `gated_engram_residual`, including when that
computation runs on the consumer GPU. It does **not** cover the fused Triton gate
selected for supported **HC=4, hidden dimension 5120** inputs. The successful small-fixture
model-hook runs cannot be reported as fused-gate or production-shape acceptance;
those remain separate tests.

For v2, the full fixture was generated **only on the Owner host**. Exactly nine
consumer-only files were copied to a fresh directory on the consumer host;
`owner_checkpoint` and embedding safetensors were not copied. The actual v2
cross-host `bootstrap_probe.py` run checked real tokenizer hash IDs against saved
CPU expectations, then checked both layers' lookup outputs against independently
generated selected-row expectations. All 6 × 2 checks matched exactly.

The v2 Owner held **2 aligned allocations totaling 4 MiB**. Consumer attach
changed current `torch.cuda.memory_allocated()` by **0 bytes**. This is not peak
memory, driver/page-table overhead, full-model capacity or total-RSS accounting;
small verification/output buffers are allowed. Consumer cleanup and process exit
were confirmed, with every native resource/failure counter zero.

### Genuine model-hook and layer-forward results

After building the main config and compute native libraries and verifying their
required imports, one consumer process ran on the Owner host and another ran on
a different host/container. These were separate runs against the **same Owner
dataset/generation and two allocations**, not a concurrent serving benchmark.
Each called real `DeepSeekV41Model._load_extra_weights()` and `_prepare_engram()`
on a small model shell; actual `Engram.from_checkpoint()` and `Engram.forward()`
then used imported lookup, `V41MXFP8Linear`, the native RTP group-32 quantization
kernel, DeepGEMM projection and the generic gated-residual branch.

Each run passed all **12 layer/case checks** and exited with code 0. Hash IDs,
masks, lookup values and WKV outputs matched exactly. The maximum final-output
absolute error was **0.0009765625**, and the maximum error/bound ratio was **0.5**,
under the predetermined two-BF16-ULP tolerance with a `1e-5` near-zero floor.
Both runs had identical aggregate metrics. External tensor pointers matched the
native mapping, repeated model loading reused the mapping, and unmanaged Graph
use was rejected. There was no consumer checkpoint table or projection shim.

These checks do not construct attention, MoE, scheduler, the complete
`DeepSeekV41Model.forward()`, or HTTP serving. Model-hook mappings remained alive
until each consumer process exited. After all consumers had exited, the v2
Owner received explicit STOP, reported `CLOSED` with every native resource and
failure counter zero, and exited. Private bundle/capability files were removed.
This does **not** mean all driver-accounted GPU memory reached zero.
A compact, machine-identifier-free result is available in
[VALIDATION_RESULTS.json](framework/VALIDATION_RESULTS.json).

### Earlier v1 coverage and correction

The v1 bootstrap/hash/lookup run also passed 12 layer/case checks, but its three
singleton decode inputs had all history slots set to `-1`. They did **not** test
rolling decode history. Its deterministic embedding rows repeated every 34 rows,
so a matching selected value could not uniquely establish the row identity.
The v2 fixture and ten new CPU tests correct both coverage gaps; v1 results must
not be relabeled as v2 or as rolling-history/all-row-identity validation.

During v1 setup a complete fixture had initially been generated on the consumer;
its task-created owner checkpoint was hash-checked and removed before GPU tests.
V1 cannot establish that no complete table ever existed there. V2 used the
Owner-only generation and restricted consumer transfer described above.

The v1 fault run intentionally changed an expected sample digest. Real GPU reads
failed admission with `GPU_READ_SAMPLE_MISMATCH`; cleanup returned every native
counter to zero and the probe process exited after verifying the expected failure.
This tests incorrect-content rejection, not hardware fault recovery, and it has
not been described as a v2 fault rerun. After all v1 consumers had exited, the old
Owner reported `CLOSED` with zero counters and exited. The separate v2 Owner was
also closed after the model-hook tests, as described above.

## Code and reproducibility

- `engram_bootstrap.py`: strict private-file bootstrap, hash ABI binding, sample
  verification and process-lifetime retention.
- `engram_model_attach.py` and `DeepSeekV41Model`: opt-in transactional model hook,
  repeated-initialization reuse and unsupported-Graph rejection.
- [Fixture setup](framework/FIXTURE.md): deterministic tokenizer and checkpoints.
- `framework/checkpoint_owner.py`: immutable allocation owner; explicit STOP only
  after confirmed consumer exit.
- `framework/bootstrap_probe.py`: the completed cross-host hash/lookup test.
- [Model-hook consumer](framework/CONSUMER.md): the completed small-fixture GPU
  test; missing real native libraries fails rather than substituting engine ops.

The publication tree preserves the exact tested Python source hashes. Its
19 model-attachment tests were also rerun locally on Python 3.10.20. Changed
Python files passed Python 3.10 syntax checks and the diff passed whitespace
checks. Public documents summarize results without copying environment logs.

New Bazel CPU targets describe the narrow dependencies for bootstrap and model
attachment tests. Both targets explicitly set `GPU_COUNT=0` so the repository
global test environment does not request a GPU lock for these CPU tests.
Both targets then ran through the **original main Bazel wrapper** on the Owner
host with `--nocache_test_results`: model attachment passed 19 tests and bootstrap
passed 29, with zero skips and no GPU probe/lock. This is one-host Bazel execution;
the corresponding BUILD change was also synchronized to the consumer.
The consumer binds its selected CUDA device immediately after importing torch,
before native/RTP dependencies can initialize CUDA on another device. Bounded
config-library checks inspect the checkout library directory and at most two
levels of `bazel-bin`; a standalone external `PYTHONPATH` entry cannot bypass
that check and trigger the RTP missing-library recursive search. These source
changes match the version tested by the 16-case harness CPU suite on both hosts.
The v2 fixture adds ten CPU cases; the combined 26-case framework suite passed
on both hosts with no skipped cases.
Native build and dependency readiness are separate from Python UT results.
Environment-specific logs, host identifiers and private capabilities are kept
outside this public repository.

## Development limits and remaining gates

The small defaults remain 1 GiB per allocation and 8 GiB total at file bootstrap;
the fixture Owner defaults to a 64 MiB loading budget. The capacity extension
adds explicit trusted-local opt-in up to 128 GiB per allocation and 256 GiB total,
while retaining 64 MiB maximum loading chunks. See [capacity policy](CAPACITY.md).
The GPU results above used the earlier small-capacity native build. A subsequent
[4.125 GiB cross-host probe](framework/LARGE_OFFSET_PROBE.md) validated the rebuilt
bridge and lookup across the 32-bit byte-offset boundary. The complete real
[188.83 GiB generation](framework/REAL_CHECKPOINT_LOOKUP.md) subsequently passed
12 exact cross-host lookup cases and consumer cleanup. That separate probe did
not run full-model HTTP. A later [40-layer TP4/EP4 eager HTTP smoke](framework/FULL_MODEL_HTTP.md)
passed for one complete four-worker RTP instance; it is recorded separately.
These ceilings do not establish available NUMA/container memory.
The VMM/native bridge and required RTP
native libraries must be prebuilt and importable before launching consumers;
bootstrap does not compile the bridge extension inside model forward. Triton
and DeepGEMM kernels may still JIT on first use and need warmup before steady-state
latency measurement. These correctness runs do not establish warmed latency.

The genuine Engram model-hook/layer-forward path is now validated for this small
eager fixture on local and remote backing. Remaining acceptance covers the
local-table numerical comparison of the complete model, production-geometry
fused-gate equivalence, and two independent RTP serving deployments under
concurrency. The separate complete-model smoke exercised the engine and all
four model ranks with the real checkpoint; it did not establish those remaining
gates.
Engine Graph, sCR, unload/reload and Owner/host failure recovery need explicit
lifetimes and additional acceptance work.

The additional **100-microsecond** target remains unpassed. These new correctness
runs make no latency claim and do not replace same-format pinned-host paired
measurements, complete-step TTFT/TPOT or intended-concurrency testing.
