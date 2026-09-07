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

`lifecycle.yaml` defines four explicit `batch-window` variants. `normal` and
`strict` each contain 59 stages and preserve the five original lifecycle
contracts below. `kv_skew_hot` and `kv_skew_cold` retain the two pilot programs
with their 2P/2D environment and independent execution budgets. The main program
uses the 2P/4D fault preset, PRIORITY/FIXED_WINDOW/BATCH axes and omitted queue
timeout. The compiler/runtime interface requires `23b3893059` or a descendant.

| Legacy contract | Actual stage/check or evidence |
| --- | --- |
| `elastic_add_flow`: new worker receives traffic within 10s | `first_traffic.received` |
| Add-flow success rate at least 90% | `add_availability.nonempty`, `.complete`, `.success_rate` |
| `elastic_add_preference`: 15s baseline, 45s post-add observation | `baseline_window`, `post_window` timestamped counters and HTTP health probes |
| Steady newcomer share at most 60% normal/loose, 50% strict | `preference_shares.new_share`; explicit normal/strict variants |
| Every old worker retains at least 10% steady share | `preference_shares.old_floor` |
| Preference flow success rate at least 90% | `preference_availability.success_rate` |
| `elastic_rebalance`: 50 requests before and after addition | `rebalance_baseline`, `rebalance_after_add`: `.complete`, `.no_errors` |
| New worker receives positive share strictly below 60% | `rebalance_share.nonempty`, `.new_share` |
| `elastic_remove_flow`: graceful removal, no request errors | `remove.membership`, `remove_zero_errors.success_rate` |
| Removed worker disappears from discovery and Master | `removed_topology.discovery`, `.master` |
| Scheduler, Prefill batch and Decode load drain | `remove_accounting.scheduler`, `.prefill_batches`, `.decode_load` |
| `elastic_add_remove_cycle`: three full cycles | Explicit `cycle1_*`, `cycle2_*`, `cycle3_*` addition, topology, received-traffic, removal and zero-error checks |
| Final request succeeds and topology returns to 2P | `recovery_completed.comparison`, `recovery_no_errors.comparison`, `final_topology` |
| `elastic_kv_skew_shrink_hot/cold` | Separate variant seed, actual baseline-hit guard, scale, transient/steady windows, recovery and fixed verdict checks |

The normal variant also represents the legacy loose share ceiling; selecting a
variant is explicit and does not silently depend on the runner's ambient grade.
Preference transient peak and steady subwindow swing remain observations, not
invented pass thresholds. Rebalance retains concurrency 10, 50 unique cold-key
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
A blocked later stage is not counted as covered by an earlier passing stage.

Local tests compile and execute both main programs and both skew variants with
simulated external engines. Negative cases cover short add-window failures,
removal errors, rebalance errors, share boundary equality, old-worker starvation
and absent Decode accounting fields. New-family real Java mock acceptance and
independent contract acceptance are pending. Historical pilot execution results
keep their original instance IDs and SHA; they are not relabelled as acceptance
of these new variants. Legacy Python functions remain available.

## Remaining families

| Final family | Remaining migration |
| --- | --- |
| `elastic_pending_drain` | Explicit pending construction and cohort ledger, legacy 40s visible terminal, accounting cleanup, new zero-error assertion |
| `elastic_concurrent_mutation` | Bounded concurrent add/remove program with the original independent robustness checks |

The later full-shrink, transient-imbalance and steady-recovery contracts still
remain legacy. Along with the two skew variants now mapped above, these are the
five additions beyond the original eight elastic cases (13 current legacy
cases). They must not be deleted to reach the four-family target. Two final
families have YAML implementations; this is not acceptance of all four.
