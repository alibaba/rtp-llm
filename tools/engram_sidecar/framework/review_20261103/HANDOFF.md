# Distributed Engram review checkpoint — 2026-11-03

## Source and scope

Read DeepSeek Harness session `session-fbf5f492-c17c-4178-895a-7e9ace5d4179`
through 2026-11-03 17:19:50 CST (64 turns, 15,396 events).
This is a review checkpoint, not a new deployment or performance acceptance.

Public branch: `codex/engram-sidecar`; inherited HEAD `91f7fda48948480fc333486f3075a23e579abb8d`.
Verified the fork remote has that commit. The session reports rebasing public and
internal branches onto `feature/dsv41-release-20261019`, not main.

## Latest user contract in the handed-over session

- RTP workers hold distributed shards themselves; no separate holder/sidecar.
- Prefill independent gangs receive a shared ordered address list.
- Decode discovers its parts via the existing gang Store; `ENGRAM_SHARD_COUNT`
  enables/checks the requested count. Device and NUMA placement remain local.
- Platform may supply the exchange port; default 29483.
- No mandatory platform-generated UUID, explicit index or ranks-per-part input.
- Startup exchanges handles; online GPU lookup does not use RPC/Store.

Keep this minimal external contract while fixing implementation correctness.
A new run identity, if required, should be negotiated internally rather than
reintroducing the rejected mandatory UUID environment variable.

## What the session actually establishes

- Earlier four-host P2 / D2 development setup: session records full-model health
  and nonempty chat responses. This predates the latest discovery changes.
- Latest ARM64 image reported by session: `rtp_llm_cuda13_arm:0.2.0_0.2.0_2026_11_03_16_05_91f7fda48_arm64_accelerated`, CI run 85789146.
- Session reports Whale `os30_engram` v2 RUNNING, but explicitly says Engram
  role environment variables were not installed and supernode membership was
  not verified. RUNNING is not evidence of shared-memory inference readiness.
- Latest full P4/D4 sharing, correct generated output, Graph and <100us decode
  overhead have not been independently established by this review.

## Local verification in this review

Ran the exchange, publish, descriptor and capacity unittest modules together:
56 tests passed after fixing one host-dependent test. The test had assumed the
local hostname resolves to 127.0.0.1; now it injects addresses for every case.
This fixes test portability, not production address discovery.

No running model was restarted or production deployment changed by this review.

## Open review findings

1. `_local_addresses()` resolves hostname IPv4 only; it does not enumerate NICs.
   Do not assert that a platform virtual IP always matches without checking it.
2. Default generation is deterministic from deployment inputs. A failed startup
   can leave Store keys; partial/repeated startup isolation needs a fresh internal
   attempt identity or a proven cleanup/lifecycle contract. Unsupported partial
   restart alone does not enforce that contract.
3. Hashing a safetensors index/config is not a content hash of weights. Different
   weight bytes can have the same index. Explicit immutable revision is still an
   option; automatic identity requires stronger evidence before production.
4. Gang discovery currently exchanges hostname-derived addresses using a fixed
   Store prefix. Check stale keys, ambiguous/multiple interfaces, topology-source
   precedence, and the multiple-instances-each-spanning-multiple-hosts case.
5. The old publication probe hashes CPU input but does not prove GPU readback.
   Add real mapped GPU reads and Graph replay for the latest self-publish path.
6. External storage currently disables async prefetch. No measured full-model
   latency acceptance exists for the latest discovery/storage version.

## Corrections to the handoff's deployment advice

- The ~98GB per-host figure came from P=2. P=4 must use the actual per-layer row
  partition and aligned allocation sizes; do not reuse 98GB as a requirement.
- NUMA IDs are local to each host. The code does not require all hosts to use the
  same ID. Whether a role-wide template can vary it is a platform question.
- CPU/GPU device accessibility, IMEX permissions and common fabric domain must
  be verified on the actual nodes; scheduler configuration alone is insufficient.
