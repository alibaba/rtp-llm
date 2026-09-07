# Priority preemption migration

Owner: agent4. Source baseline: `295af797bd7ed3a842c9cad42b5722c64cd24c9a`.
This checkpoint implements one complete candidate program out of fourteen old
contracts. No old case is called or deleted. Independent review, default catalog
integration and actual Java execution remain pending.

## Candidate: same_priority_zero_eviction

Legacy source: `flexlb_ft/cases/priority/atpm_same_priority_zero_eviction.py` and
its request, configuration and dispatch-order helpers in `support/priority.py`.
The only declared profile is single-nonbatch. Actual axes remain QUEUE,
PRIORITY, SINGLE, NON_BATCH. Environment is 1P/4D with default perf, waiting cap
8, per-worker request inflight cap 1, queue timeout 60000ms, and preemption
allowed only for PREFILL_QUEUED. No absent priority is normalized to P50;
this case explicitly sends priority=50 for each of its ten requests.

| Legacy construction / predicate | Explicit program |
| --- | --- |
| Slow the Prefill to 4000ms, synchronize 1.5s | slow, sync |
| One 2048/2 priority-50 placeholder must be admitted | placeholder, placeholder_settled, placeholder_admitted |
| Wait at most 6s for P waiting+running >=1, poll .1s | placeholder_pending, typed explicit Prefill owner |
| Eight queued priority-50 peers plus incoming 50, all 2048/2, .15s after every submission | wave, nine request records in submit order |
| All Schedule decisions settle before draining placeholder then wave | wave_settled, placeholder_drain, wave_drain |
| Schedule budget 90s, Generate RPC 120s, fresh 35s consumer drain per request | existing PriorityWave owns actual threads and RequestBatch; explicit 90/35/315 stage caps |
| PR4: no 8400/8429 among nine wave requests, all complete, same-priority FIFO dispatch shape | same_priority.PR4 |
| AT3: incoming terminal is exactly 200 | same_priority.AT3 |
| P6: wave shape, all nine complete, Master scheduler/P/D ledgers clean within 30s | same_priority.P6_terminal AND master_clean.all_owners_zero |
| Finally restore P perf, drain/cancel owned work | restore plus registered cohort/perf/environment cleanup |

Dispatch uses each wave request's actual Prefill lifecycle running_ms, then
Schedule settlement rank and wire ID for ties. The placeholder participates in
settlement ranking but is excluded from wave FIFO and zero-eviction predicates,
as in the old design-final classifier. No issue-order timestamp is substituted
for engine dispatch. Raw cohorts, pending snapshots, lifecycle snapshot,
observed order and terminal outcomes are saved as artifacts.

## Explicit execution boundaries

- RequestBatch enforces actual consumer exit and transport completion; thread
  existence or a Schedule response is insufficient evidence of completion.
- Missing/ambiguous lifecycle, missing/invalid pending counters and malformed
  owner evidence are ERROR, unlike the old helpers' sentinel/default handling.
  RPC errors and consumer drain failures propagate through the framework rather
  than being silently collected as an untyped old failure. Cleanup is mandatory.
- This candidate requires all nine wave requests to succeed. Therefore any
  failed stream fails its verdict, independently of code extraction. In-band
  CANCELLED=2 is mapped to the old 8429 family for diagnostics. General trailing
  metadata/error-family assertions for later victim programs are NOT implemented
  or claimed by this checkpoint.
- The old P6 conjunction is represented by two checks, including the separate
  30s Master clean stage. A failed earlier stage blocks dependent work, while
  cleanup still executes. There is no extra global engine-zero business check.
- Schedule workers are started asynchronously; the placeholder-settled stage
  restores its original blocking dependency. An explicit zero submission gap
  avoids inserting a .15s pause after the standalone placeholder.

## Pending contracts (13)

- atpm_preempt_prefill_queued
- atpm_preempt_decode_engine_owned
- atpm_preemption_disabled_zero_eviction
- atpm_timeout_attribution
- atpm_comparator_frozen_weak
- atpm_error_code_family
- atpm_config_strict_reject
- atpm_decode_reservation_priority
- atpm_observability_integrity
- atpm_preempt_prefill_queued_live
- atpm_preempt_decode_reserved_live
- atpm_preempt_cancel_not_found
- atpm_preempt_cancel_tombstoned

These require their own actual profile/config axes, typed terminal evidence,
owner-directed cancellation ACK and resource lifetime checks. They are absent
from the runnable YAML until their complete programs are implemented.

## Integration and verification

The dedicated action exports HANDLERS. The core integrator registers these in
the shared catalog; this branch does not edit priority.py or the catalog.
The existing priority queue test scans the entire priority directory and asserts
one plan. Its owner must scope that test to priority_queue.yaml when this new
family is integrated, keeping the original seven-request execution assertions.

`test_scenario_priority_preemption.py` loads the exact YAML through the formal
loader/compiler and executes real Schedule and Generate consumer threads with
only external IO faked. It covers complete ten-request execution, inverted
engine dispatch, missing lifecycle evidence, a yielded incoming, a rejected
placeholder blocking the wave, and compiled configuration/budget/order checks.
Local fixtures are not Java PASS or complete fourteen-contract acceptance.
