# Master migration contract

Status: all five families are implemented. Two exact batch-window lifecycle instances have real Java PASS evidence on fixed integration commit `5affc1c6de64fc8a1bf0d74dc3789d035c9a7307`, independently accepted by agent3. The other 23 profile instances have no real execution claim here. Legacy functions remain intact and are never invoked by these YAML programs. Baseline legacy source: commit `834b5eea2866dea71e3d27adb37f8c8a8b02c160`, `flexlb_ft/cases/master/`. Paths below are relative to `online_eval`.

Nine legacy cases map to five YAML families, ten variants and 25 profile instances. `all4` means batch-window, single-nonbatch, single-batch, window-nonbatch. `batch` means batch-window only. HA uses actual owned dual standalone masters, without an ambient HA skip gate, ZK forwarding or Tier-3 claims.

| Legacy case | YAML family / variant | Profiles |
|---|---|---|
| master_kill | master_lifecycle / kill_single + kill_dual_b_to_a (former explicit HA gate) | batch |
| master_freeze | master_lifecycle / freeze_short_long | all4 |
| master_quota_block | master_dispatch_quota / single_prefill_ttl | batch |
| master_coldstart_burst | master_coldstart / burst_twenty | all4 |
| master_ha_failover | master_ha_failover / standalone_a_to_b | batch |
| fallback_direct | client_fallback_failback / all_masters_down | all4 |
| failback_wraparound | client_fallback_failback / wraparound | all4 |
| fallback_negative_errorcode | client_fallback_failback / negative_errorcode | batch |
| direct_generate_error | client_fallback_failback / direct_generate_error | all4 |

Each check below is qualified as `stage.check`. Windows use the actual Java client's issue timestamps. Missing/malformed observations cause ERROR, and business predicates produce FAIL; cleanup failures cannot become findings. Control ACK alone is never fault-effect evidence.

## Old predicates to executable checks

| Variant | Original predicate | YAML check |
|---|---|---|
| kill_single | baseline request completes without error | baseline_complete.comparison; baseline_no_errors.comparison |
| kill_single | restored master discovers/alives 2P/4D | restored_topology.topology |
| kill_single | scheduler and endpoint inflight clean | restored_inflight.inflight |
| kill_single | fresh recovery request completes without error | recovery_complete.comparison; recovery_no_errors.comparison |
| kill_dual_b_to_a | steady non-failover rows >=10, all B; rescued pre-kill boundary rows excluded | steady_b.criterion |
| kill_dual_b_to_a | retry in 10s straddle, >=1 switch row A, failures <=max(1,floor(5%*N)) | retry_seen.criterion; switch_to_a.criterion; switch_errors.criterion |
| kill_dual_b_to_a | post-switch A share >=95%, success >=90%, no duplicate IDs | after_a.criterion; after_success.criterion; unique_requests.criterion |
| kill_dual_b_to_a | restarted B PID changes, topology reconverges, scheduler/endpoint inflight clean | restart_b.process_identity; ready_b.topology; clean_b.inflight |
| kill_dual_b_to_a | surviving A serves 20 serial recovery requests with >=95% success | recovery_rate.comparison |
| freeze_short_long | short-hang requests, if issued, complete on B; >=3 post-thaw burst rows all OK on B; post window stays B | short_verdict.short_hang |
| freeze_short_long | same master process survives freeze | continuity.same_process |
| freeze_short_long | discovered topology full, no regression, ready after thaw | continuity.discovered_continuity; ready_b.topology |
| freeze_short_long | if scheduler inflight was nonzero before freeze, it stays >=1 immediately after thaw | continuity.scheduler_continuity |
| freeze_short_long | long-hang judged window contains failover and at least one A target | retry_seen.criterion; switch_to_a.criterion |
| freeze_short_long | >=1 pre-freeze issued row, every row has visible success/error terminal | visible_terminal.criterion |
| freeze_short_long | >=1 deadline row in the straddling window | deadline_visible.criterion |
| freeze_short_long | post-thaw success >=90%, A share >=80%, no duplicate RIDs | post_success.criterion; post_on_a.criterion; unique_requests.criterion |
| single_prefill_ttl | all four fire-and-forget Schedule calls accepted | all_fill_admitted.all_admitted |
| single_prefill_ttl | quota actually held, scheduler inflight >=1 within 15s | quota_held.scheduler_inflight |
| single_prefill_ttl | 10 concurrent requests, failure rate >=50% | block_verdict.comparison (success <=50%) |
| single_prefill_ttl | scheduler inflight becomes zero within 95s | ttl_empty.scheduler_inflight |
| single_prefill_ttl | prefill alive again | ready.topology |
| single_prefill_ttl | 20 serial recovery requests, >=18 succeed | recovered.comparison (>=90%) |
| burst_twenty | immediate burst with stable readiness window zero; 20 requests / concurrency 10 / >=16 successes | verdict.success |
| burst_twenty | sampled final topology has alive=discovered=expected for both roles | verdict.topology |
| burst_twenty | successful requests use >=2 prefills, max share <=80% | verdict.balance |
| standalone_a_to_b | >=10 steady rows, all target A | steady_a.criterion |
| standalone_a_to_b | failover in 10s lookback + switch window; at least one switch row targets B | failover_seen.criterion; switch_to_b.criterion |
| standalone_a_to_b | switch failures <=max(1,floor(5%*N)) | switch_errors.criterion |
| standalone_a_to_b | after switch >=20 rows, B share >=95%, success >=90% | after_b.criterion; after_success.criterion |
| standalone_a_to_b | B ready and answering topology info | b_ready.topology |
| standalone_a_to_b | no duplicate request IDs | unique_requests.criterion |
| all_masters_down | >=10 steady rows, all route through master | steady_master.criterion |
| all_masters_down | outage fallback >=10 rows with >=90% success; fallback share >=80% | fallback_success.criterion; fallback_share.criterion |
| all_masters_down | outage master-route count zero; failures <=max(1,floor(5%*N)) | no_master_during_outage.criterion; outage_errors.criterion |
| all_masters_down | no duplicate RIDs, including fallback rows | unique_requests.criterion |
| wraparound | A converges, 20 fresh serial recovery requests >=95% success | ready_a.topology; recovery_rate.comparison |
| wraparound | retry seen in 10s straddle; >=1 switch request targets A | retry_seen.criterion; switch_to_a.criterion |
| wraparound | failures and business-error rows each <=max(1,floor(5%*N)) | switch_errors.criterion; no_business_storm.criterion |
| wraparound | after-switch A share >=95%, success >=90%, no duplicate IDs | after_a.criterion; after_success.criterion; unique_requests.criterion |
| wraparound | restored A scheduler and endpoint inflight clean | clean_a.inflight |
| negative_errorcode | >=5 schedule_error rows in injected business window, all contain 8431 | business_errors_seen.criterion; business_code.criterion |
| negative_errorcode | business window entirely master-routed, no fallback, failed route or failover | business_route.criterion; business_no_fallback.criterion; business_no_failed.criterion; business_no_retry.criterion |
| negative_errorcode | >=3 deadline rows, all failed-route and no retry | deadlines_seen.criterion; deadline_route.criterion; deadline_no_retry.criterion |
| negative_errorcode | deadline window contains no fallback | deadline_no_fallback.criterion |
| negative_errorcode | after 75s client finishes, A scheduler inflight <=8 within 150s | tail_settle.scheduler_inflight |
| direct_generate_error | baseline direct stream completes without error | baseline_finished.comparison; baseline_no_error.comparison |
| direct_generate_error | actual direct GenerateStreamCall errors during injection | injected_error.comparison |
| direct_generate_error | fresh direct stream completes after clear | recovered_finished.comparison; recovered_no_error.comparison |
| direct_generate_error | all prefill engine inflight zero, leak flag false after recovery | recovered_clean.engine_inflight |

## Explicit additional checks and limits

- `direct_generate_error/injected_clean.engine_inflight` independently checks engine inflight before clear, stronger than the old post-recovery-only check. It is separate from the retained old checks.
- Both kill variants retain sequential topology convergence (60s), then a separate scheduler/endpoint-clean window (10s starting after convergence). The clean stage also rechecks full topology, an additional constraint; these two deadlines must not be merged into one 60s window.
- Master restoration verifies the owned process identity. Finite master batches require consumer exit and transport-terminal records; a completed Future alone cannot pass cleanup. Cancelled jobs which never started remain explicitly NOT_STARTED, with no fabricated consumer exit.
- `master_ready` requires ready=true and both full discovered/alive counts. Where the old predicate checked only a minimum alive count or B readiness, this is stricter. Quota `ready.inflight` additionally checks scheduler/endpoint zero after restart. These extra checks are not presented as legacy-only equivalence.
- Count predicates such as “at least one switched request” use target_count>=1, not an arbitrary 1% threshold. Failure tolerance is zero for count<=1, otherwise count/N<=0.05, exactly equivalent to the original integer tolerance; empty traffic cannot independently establish success.
- Freeze scheduler continuity with zero pre-freeze inflight preserves the old conditional contract and records `nonzero_before_observed=false`; it does not prove preservation of a nonzero ledger.
- `business_code.criterion` preserves the old literal substring test on Java error text. It is not a typed or exact code assertion: text containing `84310` also contains `8431`. A strict typed-code check remains unimplemented and requires an explicit structured Java client field/protocol; it must be added as a separate check, not reported as already covered by this legacy predicate.
- The raw Java client field guard remains mandatory. Structured port ACK and owned resource handles do not prove engine generation identity. No JavaMock conclusion extends to C++ Engine slots, KV references, GPU memory or an exact global TTL.

## Local verification

Compile the shipped master YAML with the registered master, engine_control and engine_fault handlers. All 25 plans have checks, fixed resource budgets and no dynamic additions. Unit tests cover partial quota admission, one-of-1000 switch evidence, empty negative windows, strict inflight parsing, consumer-terminal cleanup evidence, owned freeze/restart, direct RPC bypass of Schedule and explicit profile/layout enforcement.

Catalog registration and integrated execution are owned by the core integrator. These local checks do not substitute for the real Java pilot or the 29-family migration acceptance.


## Fixed two-instance Java pilot (2026-09-08)

Source: `5affc1c6de64fc8a1bf0d74dc3789d035c9a7307`, clean integration tree.
Its Master YAML and action files are byte-identical to
`dab752d830fcf9a9bd886f9b280a8a1b9d208b7d`. The formal parent runner selected
exactly two instances with `--source yaml --profile batch-window --grade normal
--parallel 1 --shard case`; no profile or concurrency expansion was made.
Evidence root: `/tmp/agent1-yaml-evidence/master5aff`.

| Exact instance | Result | Elapsed | Checks | Cleanup |
| --- | --- | --- | --- | --- |
| `master_lifecycle::kill_single::batch-window` | PASS | 17453ms | 9 PASS | 4 PASS |
| `master_lifecycle::kill_dual_b_to_a::batch-window` | PASS | 108478ms | 13 PASS | 5 PASS |

`master-two/lane0/part0-yaml/scenarios.json` records 2 passed, 0 failed,
0 findings and exit code 0; the parent also exited 0. Single-Master restart
changed PID 101738 to 102157; restored topology was 2P/4D and the separate
inflight check returned zero. The two ordinary request artifacts have business
completion, consumer_done, consumer_completion_verified and nonempty
transport/consumer exit timestamps. The exact verification field is
consumer_completion_verified; no distinct completion_verified field is claimed.

Dual-Master restart changed B PID 103646 to 106151. Actual checks measured
steady B share 1.0, eight retries, 213 switch-window requests to A, zero switch
errors, after-window A and success shares 1.0, and zero duplicate request IDs.
The raw Java client log contains 1894 requests (all status ok) and eight
failovers. The separate 20-request recovery batch has success rate 1.0;
its records include consumer_completion_verified. These are different evidence
schemas and must not be conflated.

`source-jar-verification.json` records 783 matching tracked files, no source or
JAR hash mismatch, and unchanged Java/build inputs relative to the already built
9d8576 baseline. The two verified JAR SHA-256 values are
`0602971206b1fdb0b158ec7a71a8389b15fe8879b4962daeca1fb59db9938278` and
`0bdbfcf8205daed79cd985608e207679a9a2193bf5958dc670e2249e05ff3baa`.

`master-two-live-lock-audit.json` shows the parent owning both complete port
interval locks and the output-directory lock. `master-two-post-audit.json`
retains the initial transient busy port 61000; no root cause is inferred.
The later `release-audit.json` records no busy ports, all three locks reacquired
and all eight owned process IDs gone. This release JSON is an executor-produced
record; the raw tool receipt for that final port/lock recheck was not archived,
so its provenance is weaker than the retained live-lock and process receipts.
The unrelated sentinel survived scenario
cleanup. `lease-release.json` confirms lease release; the wrapper was already
PPID-1 zombie state rather than a live test process.

This pilot does not establish real execution coverage of all 25 Master
instances, all four profiles, the other Master families, or the full migration.
Legacy functions remain retained. Agent3 independently accepted these exact two
instances: it recomputed all 783 source hashes from the frozen Git archive,
compared raw results against aggregate fields, and recomputed the 1894 client
events (237 steady requests, B share 1; eight straddle failovers; 213 switch
requests to A, zero errors; after A/success shares 1; zero duplicate IDs). It
verified the retained ownership and release evidence with the provenance limit
above. The final 200-port scan range is supported by the same archived tar and
its collection command, whose hash and range(base,base+200) were checked; the
original scanning MCP envelope remains unavailable. This is selected-instance
runtime acceptance, not full-family acceptance.


`../../migration/master_runtime_coverage.json` tracks all 25 exact instance IDs separately. The remaining
23 instances are assigned to agent1 in four sequential single-lane profile groups
(8 batch-window, 5 each other profile), using the same fixed 5aff source. Assignment
and passing dry-runs are not runtime PASS. Business failures retain their original
predicates; environment or cleanup anomalies stop later groups for investigation.
