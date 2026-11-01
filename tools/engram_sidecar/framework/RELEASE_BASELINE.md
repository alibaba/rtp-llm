# Engram release baseline — 2026-10-21

Development branch: `codex/engram-sidecar-dsv41-release-20261019`.
Release base: `cb42b084a8a6350daa70897555f18d5f1b47f58e` from
`feature/dsv41-release-20261019`.
The previous `codex/engram-sidecar` branch remains at
`59df472e77091323cadbc694d6b95ca37bd707a6` for historical reproducibility.

All 19 development commits were replayed. `range-diff` maps all 19 without
dropped commits: 16 patches are unchanged; three resolve conflicts around
release model/prefetch hooks and test targets. The release's prefetch,
non-reentrant forward checks and cleanup remain in place.

## Compatibility changes

- `ec073c709b24cb2a4f4686016e1ad1b398319ddd` adds explicit backing capabilities.
  External EGM has no CPU table view and cannot use the release's CPU-prefill
  gather. Its asynchronous lookup lifetime is not yet qualified, so startup,
  warmup and submission do not enter that path. Synchronous external lookup
  and the existing process-lifetime decode Graph opt-in remain available.
  Local backing preserves the release's default behavior.
- Warmup cache keys include backing capabilities; only capable layers warm the
  optional prefetch kernel. Matching shapes do not imply matching backends.
- `full_model_http.py` explicitly disables CPU and asynchronous Engram lookup
  for **both** storage modes. This is a symmetric storage-cost experiment,
  separate from benchmarking the release's default CPU-prefill optimization.
- `e1e4f3c5946229b98d8cf4e56a6a10af8c80ecce` adapts CPU fixtures to the release
  and makes the ctime-only mutation test's filesystem precondition explicit.
  Its checkpoint-loader change is documentation only, with identical
  executable AST after excluding the module docstring.

The release implements prefetch for ordinary target prefill. It does not
provide the proposed decode/Graph prefetch optimization. See
[prefetch analysis](PREFETCH_LATENCY_ANALYSIS.md).

## CPU validation and initial failures

Tests ran in separate processes. The container used real torch
`2.11.0+cu130`, Triton `3.6.0` and safetensors `0.7.0`, with GPUs hidden.
Selected CPU-native conversion tests compiled their actual CPU implementation.
No RTP engine native library was loaded, no CUDA context was initialized,
and no RTP Python module came from outside the tested source snapshot.

| Round | Source | Coverage | Outcome |
|---|---|---|---|
| Local standard-library suites | `ec073c709` | 17 suites; 339 discovered | 309 passed; 30 torch-dependent cases skipped locally |
| Container initial CPU suites | `ec073c709` | 16 suites; 298 cases | 282 passed; 6 failures and 10 errors in 3 suites |
| Container targeted rerun | `e1e4f3c59` | external warmup 5, release prefetch 50, checkpoint 38 | 93 passed, zero skips/failures/errors |

The container includes the 30 cases skipped locally. Across both container
rounds, there are **299 distinct CPU cases** with passing applicable results;
this is not a claim that all 299 were executed in one final-commit run.
Source manifests confirmed that the rerun changes only the three affected
test files, the checkpoint module docstring and test-selection parameters.
The initial failed results are retained.

The three initial failure causes and fixes were:

1. The CPU warmup fixture hit the release's CUDA-availability early return
   under hidden GPUs. The fixture now replaces that hardware boundary while
   running the actual warmup loop, capability selection and CPU-runtime guard.
2. The release prefetch fixture replaced its package with a namespace lacking
   an import path. The new model attachment import therefore failed. It now
   preserves the real sibling-module path and executes the actual attachment
   guard; a new case verifies a failed attachment prevents prefetch.
3. The checkpoint test assumed a payload rewrite always changed ctime. In the
   tested temporary directory, 96 of 100 immediate rewrites preserved the
   complete stat token despite changed bytes; the smallest positive ctime
   increment observed was about 4 ms. Ten trials with a 10 ms delay all changed
   ctime. This observation is specific to that environment. The test now
   establishes and asserts a real ctime-only change before checking rejection,
   with a bounded retry. It still requires rejection before Owner allocation.
   Stat tokens are not a content-immutability guarantee: checkpoint inputs must
   remain immutable throughout inspection/loading.

The three rerun suites can also be invoked individually with GPUs hidden:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest rtp_llm.models_py.modules.dsv41.test.test_engram_external_warmup
CUDA_VISIBLE_DEVICES='' python -m unittest rtp_llm.models_py.modules.dsv41.test.test_engram_prefetch
CUDA_VISIBLE_DEVICES='' python -m unittest rtp_llm.models_py.modules.dsv41.test.test_engram_checkpoint
```

## Not validated by this rebase

The release changes native ABI, Graph behavior and CUDA13 ARM dependency
versions. Matching native libraries must be rebuilt before full-model startup,
cross-host GPU correctness, decode Graph replay and performance measurement.
CPU contract tests and historical main results do not establish those gates.
The additional-latency target of 100 microseconds remains unpassed; previously
observed GPU tails are preserved in the historical reports.
