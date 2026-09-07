# Elastic migration

The target is four logical families. This directory is still being migrated;
the legacy Python cases remain until independent contract acceptance completes.
The skew-shrink pilot is now an explicit variant of `elastic_lifecycle`, not an
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

## Ordered lifecycle

`lifecycle.yaml` defines five explicit `batch-window` variants. `normal` and
`strict` each contain 57 stages and preserve four original lifecycle
contracts below. The independent 10-stage `rebalance` variant preserves the fifth
contract without preference traffic warming the new worker. `kv_skew_hot` and `kv_skew_cold` retain the two pilot programs
with their 2P/2D environment and independent execution budgets. The main program
uses the 2P/4D fault preset, PRIORITY/FIXED_WINDOW/BATCH axes and omitted queue
timeout. The compiler/runtime interface requires `23b3893059` or a descendant.

| Legacy contract | Actual stage/check or evidence |
| --- | --- |
| `elastic_add_flow`: new worker receives traffic within 10s | `first_traffic.received` |
| Add-flow success rate at least 90% | `add_availability.nonempty`, `.complete`, `.success_rate` |
| `elastic_add_preference`: 15s baseline, 45s post-add observation | `baseline_window`, `post_window` timestamped counters and HTTP health probes |
| New worker receives traffic during the full post-convergence 45s window | `preference_received.received` (transient-only traffic is allowed) |
| Steady newcomer share at most 60% normal/loose, 50% strict | `preference_shares.new_share`; explicit normal/strict variants |
| Every old worker retains at least 10% steady share | `preference_shares.old_floor` |
| Preference flow success rate at least 90% | `preference_availability.success_rate` |
| `elastic_rebalance`: 50 requests before and after addition | `rebalance_baseline`, `rebalance_after_add`: `.complete`, `.no_errors` |
| New worker receives positive share strictly below 60% | `rebalance_share.nonempty`, `.new_share` |
| Remove flow reaches the new worker within 10s of its fresh accepted baseline | `remove_counter`, `remove_traffic.received` |
| `elastic_remove_flow`: graceful removal, no request errors | `remove.membership`, `remove_zero_errors.success_rate` |
| Removed worker disappears from discovery and Master | `removed_topology.discovery`, `.master` |
| Scheduler, Prefill batch and Decode load drain | `remove_accounting.scheduler`, `.prefill_batches`, `.decode_load` |
| `elastic_add_remove_cycle`: three full cycles | Explicit `cycle1_*`, `cycle2_*`, `cycle3_*` addition, topology, received-traffic, removal and zero-error checks |
| Final request succeeds and topology returns to 2P | `recovery_completed.comparison`, `recovery_no_errors.comparison`, `final_topology` |
| `elastic_kv_skew_shrink_hot/cold` | Separate variant seed, actual baseline-hit guard, scale, transient/steady windows, recovery and fixed verdict checks |

The normal variant also represents the legacy loose share ceiling; selecting a
variant is explicit and does not silently depend on the runner's ambient grade.
Preference transient peak and steady subwindow swing remain observations, not
invented pass thresholds. Rebalance runs in its own fresh environment: baseline batch, add and topology
convergence, then the immediate post-add batch with no intervening warmup. It
retains concurrency 10, 50 unique cold-key
requests per batch, 2048 input / 2 output tokens, and separate Schedule/stream
budgets. Removal accounting retains its 95s drain budget and checks each owner;
missing owner fields are ERROR rather than zero.

Combining flows changes construction: add availability is computed only from
requests issued between one second before the add call and one second after
first observed new-worker traffic. Earlier baseline successes cannot dilute a
failure in that cohort. This reuses the longer preference flow rather than
restarting an identical short legacy flow; independent acceptance must review
that adaptation. Cohort completeness, exact topology and explicit consumer
termination are additional construction checks. Removal and each cycle keep
independent zero-error checks; the preference 90% floor cannot hide their errors.
Cycle construction also merges the old traffic-pumping phase and removal flow:
its zero-error assertion covers the entire combined flow, including pre-removal
traffic. That wider denominator is an explicit **new assertion**, not an exact
replacement for the legacy removal-only success scope. The legacy function stays
available pending acceptance of this stronger construction.
A blocked later stage is not counted as covered by an earlier passing stage.

Local tests compile and execute both main programs and both skew variants with
simulated external engines. Negative cases cover short add-window failures,
removal errors, rebalance errors, share boundary equality, old-worker starvation
and absent Decode accounting fields. New-family real Java mock acceptance and
independent contract acceptance are pending. Historical pilot execution results
keep their original instance IDs and SHA; they are not relabelled as acceptance
of these new variants. Legacy Python functions remain available.

## Concurrent mutation

`concurrent_mutation.yaml` implements `elastic_concurrent_ops` in the final
`elastic_concurrent_mutation` family. Four real threads share a candidate pool:
one Prefill adder, one Decode adder and two independently seeded removers. The
10s mutation admission window and the original add intervals (.25/.40s), remove
intervals (.40/.55s), graceful removal and serial health request followed by a
Master HTTP probe are preserved. Concurrent removal races and failed mutation
attempts remain recorded observations, not an invented zero-error contract.

| Legacy contract | Actual stage/check |
| --- | --- |
| Every sampled Master endpoint returns HTTP 200 | `crossfire.master_http` |
| Nonempty health requests, success rate at least 50% | `health.nonempty`, `health.success_rate` |
| Discovery parses after the storm | `discovery.parsable` |
| Prefill and Decode discovery counts equal mock services | `discovery.counts_match` |

Boundedness is explicit: the two adders can attempt at most 40 and 25 additions
(the maxima implied by the 10s window and original intervals), so the compiler
reserves 65 dynamic additions. Even failed attempts consume that budget. No
ordered loop replaces concurrent mutation. Each attempt retains its request,
response or exception and timestamps; health request outcomes are retained too.

`crossfire.workers_finished` and `health.complete` are additional construction
checks. Independent done events replace the old unchecked 15s thread joins;
final discovery is read only after every mutation worker exits. Slow graceful
calls may finish after the 10s admission window, within the 120s stage budget. HTTP client timeouts preserve the legacy
10s add / 95s graceful-remove budgets, clipped only by the remaining stage
deadline; 120s covers a removal begun just before the 10s admission cutoff.
Cleanup cancels request work and waits for mutation workers before tearing down
the instance-owned environment. The legacy function remains until independent
acceptance; real Java mock acceptance of this new family is pending.

## Pending drain

`pending_drain.yaml` implements the fourth logical family as two explicit programs:
`legacy_terminal` (12 stages, 12 checks) and `zero_errors` (13 stages, 15 checks).
Only the former maps to `elastic_remove_pending_drain`; the stronger variant has
an empty legacy mapping so it cannot be counted as old-contract acceptance.
Both use a fresh private 2P/2D fault environment, PRIORITY/FIXED_WINDOW/BATCH,
two Prefill batch leases and omitted queue timeout (the Java default remains).

| Legacy contract | Actual stage/check or evidence |
| --- | --- |
| Both prefills slow to 8000ms, 1.5s perf synchronization | `slow_both_prefills`, `perf_sync` |
| Serial Schedule submissions, concurrent stream consumers, 300ms accepted-request spacing | `wave`; at most 14 attempts, target four victim-routed, 1024 input / 2 output / three unique cold keys |
| At least three victim-routed requests and aggregate pending estimate at least one | `wave.victim_routed`, `.pending_nonempty` |
| Graceful removal starts the terminal clock | `remove`: 60s server drain, 95s HTTP client cap, captured timestamp immediately before HTTP |
| Every victim-routed request completes or explicitly fails within 40s | `visible_terminal.victim_visible_terminal` |
| Accounting returns to zero within 50s after collection | `accounting.scheduler`, `.prefill_batches`, `.decode_load` |
| Survivor recovery at least 19/20 | `recovery.complete`, `.success_rate`; concurrency 10, 2048 input / 2 output / three cold keys, Schedule 30s and stream 15s |
| Victim absent from mock services and file; Master sees one alive prefill | `remove.membership`, `topology.discovery`, `.master` |

`wave.no_completed_interference` is an additional construction guard: the victim's
engine completion counter must not advance and no victim client stream may have
terminated before removal. Otherwise completed requests could inflate the old
`routed - waiting - running` pending estimate. Missing or non-monotonic completion
counters are ERROR. This remains an aggregate inference, not proof of the exact
request IDs in the Master's WorkerBatcher queue. Construction and cleanup records
are stored separately so cleanup cannot overwrite the pre-removal evidence.

All Schedule attempts remain in the client ledger. Rejections and survivor
outcomes retain their legacy observation scope; they are included in the stronger
`all_issued_zero_errors` denominator. The old visible-terminal verdict is victim
only and permits explicit business or transport errors. An empty close, missing
terminal, collector timeout or collector cancellation cannot satisfy it, even
when cancellation itself ends the RPC. Fast (<=5s), stale-window (<=16s) and slow
failure shapes remain observations. `drained=true` never substitutes for either
client-visible completion or zero errors.

Every issued request owns a consumer and independent done event. Schedule and
stream budgets remain 30s/60s (90s total per wave request); the wave has a finite
450s stage budget covering the old maximum 14 serial Schedule attempts. Consumers
may span subsequent stages as registered resources. Collection retains per-stream
45s waits, records any cancellation and proves consumer exit; the 100s collection
budget covers the remaining bounded request lifetimes. Cleanup cancels and waits
for these consumers before environment teardown. Accounting starts after collection,
not 50s after removal. The dedicated recovery reuses the bounded cold batch helper;
rebalance remains exactly 50 requests with a separate 100% success contract.

Local tests use formal loader -> compiler -> runtime, real request-consumer
threads, and simulated HTTP/gRPC services. They cover legacy explicit-error PASS
versus zero-error FAIL, Schedule-rejection accounting, false pending caused by
engine completion, the 40s terminal boundary, empty/cancelled terminals, recovery
19/20 versus 18/20, and missing owner counters. Real Java mock and independent
contract acceptance remain pending; legacy code stays available.

## Remaining variants

Full-shrink, transient-imbalance and steady-recovery contracts remain legacy and
must be folded into the four families. Along with the two skew variants mapped
above, these are the five additions beyond the original eight elastic cases
(13 current legacy cases). They must not be deleted to reach the four-family
target. All four final families now have YAML candidates; this is not complete
coverage or acceptance of all 13 legacy contracts.
