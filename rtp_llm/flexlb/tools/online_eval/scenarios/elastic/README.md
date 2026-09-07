# Elastic migration

The target is four logical families. This directory is still being migrated;
the legacy Python cases remain until independent contract acceptance completes.
`kv_skew_shrink.yaml` is a pilot to fold into the lifecycle family, not an
additional final family. No production failure is converted into a finding.

## Added worker fault

`added_worker_fault.yaml` implements `elastic_added_worker_fault` as one explicit
program for the `batch-window` profile, preserving the 2P/4D fault preset,
PRIORITY/FIXED_WINDOW/BATCH axes, four batch leases and omitted queue timeout.
The compiler supplies FIXED_WINDOW and BATCH from the profile. The core request
shape/timeout interface requires commit `23b3893059` or a descendant.

| Legacy `elastic_stop_after_add` contract | Actual stage/check |
| --- | --- |
| Add succeeds and the new engine exists | `add.membership` |
| Discovery entry appears within 10s | `added_topology.discovery` |
| Master sees the third alive prefill within 30s | `added_topology.master` |
| New engine receives traffic within 15s | `first_traffic.received` |
| Stop makes alive count fall to two within 30s | `stopped_topology.master` |
| A request succeeds while the engine is stopped | `survivor_completed.comparison`, `survivor_no_errors.comparison` |
| Restart restores three alive prefills within 30s | `restored_topology.master` |
| Restarted engine accepts fresh traffic within 20s | `resumed_traffic.received` |

The restarted-worker check freezes a fresh accepted counter before starting its
probe flow. Pre-stop traffic cannot satisfy this check. The previous implementation
also took a new baseline inside its traffic-pumping helper.

Additional construction checks require exact discovered counts and preserve the
stopped engine's discovery entry. These are explicit stricter topology checks,
not replacements for the legacy alive assertions. Stop/start acknowledgements
and snapshots are saved by `engine_control`; they do not prove request success.

Probe flows preserve serial completion followed by a 200ms interval, separate
30s Schedule and 10s stream deadlines, and unique cold keys. They retain every
attempt and have separate artifact files. The legacy probe loops did not assert
a success-rate threshold; this program does not invent one. The downtime request
has its own business-success checks. Cleanup is owned by this isolated instance.

Local compiler/runtime tests exercise the entire YAML, including absent resumed
traffic and a failing survivor request. Real-engine acceptance is pending. The
legacy function is retained.

## Remaining families

| Final family | Remaining migration |
| --- | --- |
| `elastic_lifecycle` | Ordered add/remove, three explicit cycles, original 50-request rebalance batches, 15s/45s preference windows |
| `elastic_pending_drain` | Explicit pending construction and cohort ledger, legacy 40s visible terminal, accounting cleanup, new zero-error assertion |
| `elastic_concurrent_mutation` | Bounded concurrent add/remove program with the original independent robustness checks |

The five later contracts (full shrink, two skew shrink variants, transient
imbalance and steady recovery) remain accounted for separately from the original
eight elastic cases. They must not be deleted to reach the four-family target.
