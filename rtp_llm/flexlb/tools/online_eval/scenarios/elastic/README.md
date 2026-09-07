# Elastic migration

The target is four logical families. This directory is still being migrated;
the legacy Python cases remain until independent contract acceptance completes.
The skew-shrink pilot is now an explicit variant of `elastic_lifecycle`, not an
additional final family. No production failure is converted into a finding.

## Added worker fault

`added_worker_fault.yaml` implements `elastic_added_worker_fault` as one explicit
program with explicit variants for all four profiles, preserving the 2P/4D fault
preset, PRIORITY ordering, profile-selected decision/dispatcher and omitted queue
timeout. The corrected serial probe restores legacy issuance-window semantics
and the cross-restart cumulative counter check for BW and the three new profiles.
See [MIGRATION_ADDED_WORKER_PROFILES.md](MIGRATION_ADDED_WORKER_PROFILES.md) for
the correction, declared strengthening, fixtures and catalog integration.

| Legacy `elastic_stop_after_add` contract | Actual stage/check |
| --- | --- |
| Add succeeds and the new engine exists | `add.membership` |
| Discovery entry appears within 10s | `added_topology.discovery` |
| Master sees the third alive prefill within 30s | `added_topology.master` |
| New engine accepts traffic from the 15s issuance window | `first_traffic.received` |
| Stop makes alive count fall to two within 30s | `stopped_topology.master` |
| A request succeeds while the engine is stopped | `survivor_completed.comparison`, `survivor_no_errors.comparison` |
| Restart restores three alive prefills within 30s | `restored_topology.master` |
| Restarted engine accepts fresh traffic from the 20s issuance window | `resumed_traffic.received` |
| Post-restart cumulative accepted exceeds pre-stop count | `cross_restart_growth.increased` |

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

`lifecycle.yaml` defines seven explicit `batch-window` variants. `normal` and
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

The existing `batch-window/default` instance is preserved. Explicit
`single_batch`, `single_nonbatch`, and `window_nonbatch` variants now cover the
other three legacy profiles, with an additional expected-protocol check.
See [MIGRATION_CONCURRENT_PROFILES.md](MIGRATION_CONCURRENT_PROFILES.md) for
configuration equivalence, driver fixtures, and the pending acceptance boundary.

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

`pending_drain.yaml` implements the fourth logical family with profile-scoped
programs: batch-window `legacy_terminal` (12 stages,12 checks), its stronger
unmapped `zero_errors` (13 stages,15 checks), and the separate single-batch
`single_batch_terminal` (13 stages,13 checks). Both terminal variants map to
`elastic_remove_pending_drain` in their own profile; no other elastic profile is
expanded. All use a fresh private2P/2D fault environment, PRIORITY ordering, BATCH
dispatch, two Prefill batch leases and omitted queue timeout. The decision is
FIXED_WINDOW for batch-window and SINGLE for single-batch; it is not pinned by
an override. See [the single-batch migration analysis](MIGRATION_PENDING_SINGLE_BATCH.md)
for the complete configuration comparison, singleton batch-credit model,
additional FetchResponse guard and remaining Java evidence gap.

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
19/20 versus 18/20, and missing owner counters. Independent static review passed
at `33f9d1d1e0f7f64af5e9af2691e55435fd984034`, for the batch-window legacy
terminal candidate only. Real Java acceptance remains pending; legacy code stays
available. The separate single-batch candidate passed independent static review
at `df767a6e93af21cc4a436adc34e7b4ee725446a6` on core4e2139 with95 elastic
tests rerun independently. Its real Java acceptance remains pending.

## Steady recovery variant

`elastic_lifecycle::steady_recovery::batch-window` folds the old
`elastic_steady_state_recovery` into the lifecycle family as a separate 14-stage,
15-check program. It uses a private 2P/4D environment, PRIORITY/FIXED_WINDOW/BATCH
and omitted queue timeout. The legacy fingerprint-only environment marker is not
needed for the scenario runtime's instance-owned environment. This candidate is
batch-window only, matching this particular legacy case's only declared profile.
The single-batch gap described for pending drain does not apply to this case.

The serial pump preserves 2048 input / 2 output / one unique cold key, Schedule
30s and stream 30s, followed by a 200ms pause. It runs through a 20s baseline,
graceful removal of `decode-0` (60s drain / 95s HTTP), 20s transient wait,
convergence to three Decode workers, a pure 60s steady window and 20-request
recovery. Inflight-clean is not a settle prerequisite while this pump is running.
`settled_topology` is an explicit **new assertion**: the old case recorded the
post-remove `alive_ok` result in its detail but did not combine it into the
returned verdict; even `alive_ok=false` could continue into W_ss and pass. The
new program requires exactly three discovered/alive Decode workers and victim
absence before W_ss. Snapshot membership and discovery checks are also stronger
than that legacy diagnostic. These construction guards are not claimed as exact
old-contract equivalence.
The recorded pump success rate remains an observation; request consumer completion
and cleanup are still required. Stopping the pump is explicit before final verdict.

| Legacy steady contract | Actual stage/check or evidence |
| --- | --- |
| Nonempty baseline Decode traffic | `baseline_guard.nonempty_decode_traffic` |
| Last-third share max <= max(baseline+0.10, 1/3+0.15) and min >= 0.10 | `verdict.share_max`, `.share_min`, `.nonempty_decode_traffic` |
| Last-third Decode survivor waiting peak <= 2 | `verdict.waiting_peak` |
| Last-third occupancy spread <= baseline+0.05 and each peak <= 0.95 | `verdict.occupancy_spread`, `.occupancy_peak` |
| Two time-adjacent 3s windows deviate in the same direction beyond +/-0.10 from 1/3 | `verdict.oscillation`; 20 subwindows over the full steady minute |
| Recovery >=19/20 with concurrency 10 | `recovery.complete`, `.success_rate` |
| Swing, execution-time CV, cluster hit rate, Decode generate-TPS ratio and reference bands | Verdict artifact observations, never pass thresholds |

Decode shares use completed counters, not Prefill accepted counters. Empty
traffic subwindows keep their time indexes and cannot join separated departures
into a false adjacent pair. Required engine series, missing occupancy fields,
counter resets and gaps over 2.5s are errors rather than fabricated zero values;
this explicit data-completeness guard strengthens the old best-effort sampler.
Optional observation fields retain an unavailable reason. Baseline/steady windows
and removal have independent artifacts; continuous sampling spans the blocking
removal. No production code changed and no legacy function was removed.

Local tests execute the formal YAML with actual pump threads and simulated
services. Independent checks cover a balanced last third with earlier persistent
drift, queue depth three, missing occupancy, counter resets and empty subwindow
gaps. Independent static review passed at `a4ec79ebe5acccf647ae152fb388daab010f4ade`
plus `84e6250228bf1777980590a779d07b4706a6b300` on core23. Real Java mock
acceptance remains pending.

## KV-full shrink variant

`elastic_lifecycle::kv_full_shrink::batch-window` preserves both drain branches
of `elastic_kv_full_shrink` in one ordered 36-stage, 33-check program. Its private
2P/2D environment declares `decode_cache_blocks: 24`. Both initial and dynamically
added Decode workers use that actual pool size; no artificial KV-pressure setter
is used. The candidate requires the core's explicit cache-pool environment fields.

| Legacy full-shrink contract | Actual stage/check or evidence |
| --- | --- |
| Real running leases saturate the next-request Master KV gate | `fill_ok.saturated`, `fill_timeout.saturated`: 24 blocks, >=2 reserve blocks, running>0, projected next 2048 tokens strictly exceed 90% |
| Fill shape does not grow from two to three blocks on first decode step | 2035 input + 13 output, two unique cold keys per request; <=60s fill, eight serial Schedule submissions per 100ms round |
| Graceful branch actually drains | `terminal_ok.drain_branch`: `drained=true` |
| Graceful admitted requests succeed or are pre-event 8211 fill refusals | `terminal_ok.terminal_family`, `.retirement_contract` |
| Every admitted post-event terminal is <=40s; no hang | Each branch's `terminal_40s`; client timestamps, not collector order |
| Accounting cleans within 50s after branch-one collection | `accounting_ok` owner-specific scheduler/Prefill/Decode checks |
| Timeout branch is real and uses milliseconds | `terminal_timeout.drain_branch`: `drained=false`, 5000<=drain_ms<=10000 |
| Timeout yields at least one exact Decode generation retirement | `terminal_timeout.retirement_contract`, `.terminal_family`: 8510 plus `Decode endpoint generation retired`; unrelated errors fail |
| Survivor transient occupancy <=0.95 and reject delta <= ceil(max(0,victim occupied-survivor free)) | `transient_ok`, `transient_timeout`, each over the full 20s from removal |
| Branch-one steady Decode occupancy spread <= baseline+0.05; waiting peak <=2 | `steady_bounds.occupancy_spread`, `.waiting_peak` over the last third |
| Steady Decode share max <= max(baseline+0.10,0.65), min>=0.10 | `steady_bounds.share_max`, `.share_min`, `.nonempty_share`; completed-counter deltas over the whole steady window, matching the legacy implementation |
| Final survivor recovery >=19/20 | `recovery.complete`, `.success_rate`, concurrency 10 |
| Steady cluster TPS | Observation in the steady artifact, never a pass threshold |

The two background windows preserve serial 2048/2/one-key requests followed by a
500ms pause, Schedule 30s and stream **10s**. They stop for remove/accounting; an
always-running flow would prevent the zero-inflight measurement. Explicit start
anchors retain the baseline ramp and the branch-one steady ramp in the old
measurement ranges. The steady tail begins 40s after its settle anchor. Its final end and samples
are captured after `steady_stop`, including stop/drain-period completions and
waiting peaks as in the legacy case; the baseline still ends before its stop.

The candidate intentionally preserves the old literal `/set_perf` order, but
that order does **not** prove a victim-only 1000x tail. All mock services share
one `MockPerformanceModel`, including dynamic engines; branch two writes
`victim=1000`, then `survivor=60`, leaving both at 60. Its actual drain response
and client terminal checks still gate the verdict. A fixture may exercise the
timeout branch with remaining work at 60x, but that is not a Java proof of the
old comments' independent per-worker slowdowns. The fixture models the shared
field and checks every write in order. No Java or production code is changed.
See [the full-shrink migration analysis](MIGRATION_FULL_SHRINK.md) for the exact
control sequence, affected owners, and a separately proposed correction; the
legacy-mapped candidate does not silently substitute that correction.

Additional construction guards are explicit: all admitted consumers must exit;
8211 is exempt only when terminal **before** removal (the old code allowed 8211
without checking that comment's pre-event qualification); required metric series
cannot be missing; post-add topology is a hard gate (the old `v2_topo_ok` was
recorded but unused by its final verdict). The second transient verdict waits for
its entire 20s window rather than inspecting whatever partial samples happened
to exist immediately after collection. These strengthen measurement validity and
are not advertised as exact legacy behavior. Schedule refusals remain recorded
observations, outside the admitted-stream terminal-family contract. A transport
error or collector cancellation cannot impersonate a permitted business error.

Local tests execute the complete formal YAML with actual fill/pump threads and
simulated request/metric services, plus boundary tests for saturation, reserve,
zero rejection budget, milliseconds, 40s, pre-event 8211 and retirement messages.
Independent static review passed at `9520f72d8703d1bc49a03163f3ed2a00b09a1e04`
plus `c879d43dbb29ab6945a3b91b50189cf9a5ebd4c2` on core23, including the
stop-period negative fixtures. Real Java acceptance remains pending. Legacy code is retained;
a later blocked branch is not counted as covered by earlier successful checks.

## Transient imbalance

`elastic_lifecycle::transient_imbalance::batch-window` migrates
`elastic_transient_imbalance_bound` as an explicit 19-stage, 20-check program.
The private 3P/2D environment preserves fault performance, priority ordering,
queue timeout 60000ms, Master per-Prefill waiting capacity64, and the **born**
Prefill `max_waiting_batches: 16`. The latter uses the core's typed
`prefill_max_waiting_batches` field, not a post-start performance update.
Decode concurrency remains the Java default128. The full resolved Master config,
performance dictionary, topology and both pool sizes are compared with the old
spec in a local test. The unused `FLEXLB_FT_SPEC_ID` fingerprint marker is omitted
because the instance owns a private environment; it was not a runtime control.
This variant requires shared-core commit `4e2139c1398f1ed3039d235d2f736456612fa0f8`
or its integrated equivalent for the typed birth field.

The serial pump uses 2048 input, 2 output, one unique cold key, Schedule30s,
stream30s and a100ms pause. A30-request crossing burst uses concurrency15 and
stream45s (Schedule30s); every issued attempt retains its raw record and consumer
exit evidence. The removal targets Prefill1 in abrupt mode with a5s HTTP cap. Its artifact
freezes burst request records immediately before the call: starting the burst
thread alone does not prove any victim-routed request is pending, and no such
unmeasured overlap is claimed.
The transient window starts immediately before that control call and ends20s
**after the burst settles**, preserving the legacy implementation's variable
window, rather than relabeling it as exactly20s. The steady window starts after
Master convergence, spans60s, and its tail starts at+40s. The pump remains active
through recovery20 (concurrency10, >=19 successful).

| Contract | Explicit check / source |
| --- | --- |
| Surviving Prefill waiting peak <=16 | `transient_bounds.prefill_waiting`; engine metric counts requests although the configured capacity counts batches. This legacy unit mismatch is retained, not silently multiplied by batch size. |
| Decode waiting peak <=128 | `transient_bounds.decode_waiting`; Java default concurrency. |
| Master Prefill survivor inflight requests <=64 | `transient_bounds.master_inflight`; the endpoint registry's HTTP addresses select P0/P2 explicitly. This is an inflight proxy, including dispatched work, not a pure waiting counter. The victim's HTTP address is excluded. |
| All four survivors' occupancy <=0.95 | `transient_bounds.occupancy`; each engine's own pool. |
| All-survivor rejection delta bounded by displaced demand | `transient_bounds.rejects`; ceil(max(0, victim occupied blocks - sum(all four survivor free blocks))) from the pre-burst snapshot, compared with all-survivor lack-memory + admission-failure counter deltas. |
| Prefill steady share max <=max(base+0.10,0.65), min>=0.10 | `steady_bounds.share_max`, `.share_min`; accepted-counter deltas over the whole60s, matching the legacy code. |
| Steady tail waiting <=2 and P/D occupancy spreads <= respective baseline+0.05 | `steady_bounds.waiting`, `.prefill_spread`, `.decode_spread`; independent owner planes and +40s tail. |
| No survivor-routed failures | `locality.survivor_failures`; nonempty RPC address different from victim counts as survivor; empty string/None failures remain explicit unrouted observations. |
| All issued consumers exit; recovery >=19/20 | `burst_settled.complete`, `locality.complete`, `recovery.complete`, `.success_rate`. |

Transient cluster TPS and baseline*4/5*0.85, plus Decode steady share, remain
observations. Missing required metrics, endpoint identity ambiguity, counter
reset, acquisition errors and uncovered sample gaps are errors, never zeroes.
Raw continuous mock and Master samples are separate artifacts; request outcomes
do not imply scheduler, Engine-slot or KV ownership release.

Construction changes are explicit. Post-removal topology is a hard check; old
`alive_ok` was diagnostic only. An empty baseline share is an error instead of a
zero fallback. Locality now freezes the **whole stopped pump cohort after the
recovery batch**, whereas the old code inspected the pump before recovery and
could omit an outstanding consumer. Recovery requests retain their separate
>=95% contract; the enlarged pump cohort has the same zero survivor-failure
bound. Cancellation cannot be counted as successful business completion.
The old case's final best-effort inflight cleanup was not a verdict; this
candidate uses private environment teardown and does not claim a new accounting
PASS. Core first-failure blocking is explicit: blocked later checks do not count
as coverage.

Local tests use actual pump/burst threads with simulated RPC and sampler data;
they exercise the full formal YAML and boundary/error cases. They are not real
Java evidence. Independent static review passed at
`d1f84c134eca2e62a59066058b7b1bd85de108ce` on core4e2139, with90 elastic
tests rerun independently. Java acceptance remains pending.

## Coverage status

All13 current legacy elastic IDs now have batch-window YAML candidates within
the four target families. Candidate mapping is not complete profile coverage or
acceptance: the legacy implementations remain, required non-batch-window
profiles still need dedicated migration/validation, and static/model results
cannot substitute for real Java runs.
