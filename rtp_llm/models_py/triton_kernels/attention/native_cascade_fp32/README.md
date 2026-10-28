# MI308 cascade FP32 prefix

This module packages the measured RN/FP32-output prefix independently of AITER's
normal BF16-output module. It is enabled with `RTP_ROCM_CASCADE_FP32=1` together
with `ENABLE_DECODE_CASCADE_ATTENTION=1`. The FP32 mode is opt-in.

The native route supports gfx942, BF16 Q/K/V, Q/KV heads 16/8, D128, shuffled
page16 KV, non-causal shared prefixes, and query tiles of at most 128 rows. Only
verified complete shared pages enter this prefix. The short private tail uses
common softmax normalization, vectorized V loads and one wave; invalid V slots
are zeroed before multiplication. Dynamic long or unshared tails use the generic
kernel. Unsupported native geometries and KV spans at or above 2 GiB use the
FP32 Triton prefix; 64-bit tail indexing remains independent of native selection.

`beam_search_cascade_native.py` loads the uniquely named extension during
operator construction, before Graph capture. On a cache miss, AITER's build
utility compiles these nine pinned translation units with RN flags scoped to
this module. No installed source, default AITER module or global rounding
environment variable is modified. The first build needs the matching AITER/CK
headers and ROCm compiler and takes approximately one minute on the tested host;
it is excluded from steady-state request timing. A writable `AITER_JIT_DIR` may
be selected with AITER's normal cache configuration.

The four generated kernel instances preserve the upstream M64/M128 selector and
bload/gload recipes. `provenance.json` records source and dependency hashes;
`LICENSE` and original source copyright notices are retained. If precision,
ABI, source or dependency contracts change, update the provenance and module
version in the loader, then repeat validation. This source integration has been
validated with a local Bazel-owned service; a released wheel/image is a separate
packaging acceptance step.

Production entry points:

- `../beam_search_cascade.py`: request-aligned planning, stable workspaces and routing.
- `../beam_search_cascade_native.py`: independent extension loading and API binding.
- `../beam_search_cascade_tail.py`: guarded short-tail implementation.
- `../beam_search_cascade_tail_candidate.py`: compatibility imports for archived experiments.

The partial-page 714-token and alternate tile-selection experiments are not
enabled by this mode. The shared cut remains data-dependent and page-aligned
(704 tokens for the captured 714-token history).

Validation entry points:

- `//rtp_llm/models_py/triton_kernels/attention/test:test_beam_search_cascade`
- `//rtp_llm/models_py/triton_kernels/attention/test:test_beam_search_cascade_mkt`
- The `address_boundary`, `tail_safety`, `tail_dispatch` variants of that target.
- `test_beam_search_cascade_mkt_real_fixture`, selected with unittest
  `-k test_integrated_fp32_route` and `RTP_ROCM_CASCADE_FP32=1`.
- `//rtp_llm/test/smoke:rocm_mkt_cascade_e2e`, with private checkpoint/request/env
  paths supplied via `MKT_FIXTURE_*`, `MKT_E2E_MODE=off|legacy|optimized`, fresh
  `MKT_E2E_OUTPUT_DIR`, and explicit server/client CPU lists. It uses the standard
  smoke server manager, includes full HTTP responses, and records failures and
  raw latency samples. Run through the prechecked GPU-locking test workflow.

Real fixture gates use the original tolerances and compare Graph replay against
its own eager execution. These attention checks do not establish identical
whole-model beam candidate sets or business relevance quality.
