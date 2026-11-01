# Validation record: component checkpoint

Base: public main `492ecf45b3ef55b22fd4882622abfdb2e2f664f2`.
Publication preparation: 2026-10-16.

## Checks run on the published source snapshot

The following **68 CPU tests** passed on Python 3.10.20. They also passed on
Python 3.9.6. These are the same 68 cases, not 136 distinct cases.

| Suite | Cases | What it establishes |
| --- | ---: | --- |
| Portable build paths | 5 | Checkout-relative source discovery; user cache and explicit overrides |
| Manifest | 19 | Complete identity/layout/bounds validation and strict input handling |
| Sessions | 19 | State-machine races, identities, idempotency and no timeout-based reclamation |
| Storage | 15 | Import/verification and Graph-resource contracts using resource-accounting doubles |
| Benchmark contract | 10 | Paired statistics, sample/order handling and metadata validation |

All 33 changed/new Python files parsed using Python 3.10 syntax; `git diff
--check` passed. CPU doubles do not prove GPU drain, CUDA Graph or cross-host
memory visibility. No GPU test was rerun as part of publication preparation.

The publishing snapshot changes six development-tool files from the earlier
GPU-tested snapshot: repository defaults are discovered from `__file__`, cache
defaults use the user's home directory, and `ENGRAM_QUANT_BUILD_DIR` can override
the quant probe build directory. Native diagnostics and example paths were made
portable. These edits do not change allocation, lookup or numerical algorithms.
The dependency lock preserves the tested version/wheel hash while removing the
environment-specific download URL. The five new path tests exercise the changed
build-default behavior. The source
algorithms below retain their earlier component results; those results must not
be described as a fresh GPU run of this commit.

## Historical GPU component results

The pre-publication development snapshot was tested in CUDA 13 ARM64 containers
on a supported NVLink fabric, with torch 2.11.0+cu130 and Triton 3.6.0.
Environment-specific raw logs, private Fabric capabilities and machine/user
identifiers are intentionally not published.

- Native HOST_NUMA/FABRIC export/import, READ mapping and ATen external view
  checks passed on the tested platform. Native tests included 100 ordered
  attach/view/drain/close cycles and constructor rollback paths.
- One owner published the same two small layer allocations to two consumers:
  one on another GPU of the same host/container, the other on a different host
  and in a different container. Both used real RTP lookup and guarded Graphs.
- Each consumer passed 8 eager cases and 10,000 Graph replays; 10,240,000 BF16
  output values per consumer matched independent expectations exactly. Ordered
  cleanup returned the implementation's native counters to zero.
- The standard safetensors loader was exercised with deterministic real files,
  including preflight and failure paths. This is not a full production model
  checkpoint or capacity test.
- Canonical-layout work passed 160 numerical cases, 9 format-rejection cases
  and 8 real DeepGEMM projection cases on the tested configuration. Canonical
  and raw representations were both allocated for experiments; that test setup
  is not evidence of production one-copy memory use.

## Unpassed gates

The additional **100 microsecond** latency target has not passed. One historical
L=8/M=32 component run observed a maximum GPU paired delta of 1,763 microseconds
and a maximum consumer-wall paired delta of 3,170 microseconds. Both local and
remote paths had spikes; these observations do not establish their root cause.
They are component measurements, not full-model TTFT/TPOT or a whole-step SLA.

The following remain pending:

- Complete framework/engine bootstrapping, all required model layers/ranks,
  engine-owned Graph teardown and real chat/HTTP serving.
- A full real Engram checkpoint at production capacity with a verifiable
  one-copy backing ledger and multiple independent RTP instances.
- Whole-step and full-model tail latency under intended concurrency and EP
  traffic, compared with the local pinned-host baseline.
- Owner/host/GPU/fabric failure handling, sCR/restore and production service
  lifecycle, authentication and deployment integration.

Raw views or mock-based tests must not be treated as proof these gates passed.
The next checkpoint should add framework integration evidence without silently
weakening any of these acceptance criteria.
