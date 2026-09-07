# Priority preemption migration

Owner: agent4. Source baseline: `295af797bd7ed3a842c9cad42b5722c64cd24c9a`.
This checkpoint implements four complete candidate programs out of fourteen old
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

## Pending contracts (10)

- atpm_preempt_decode_engine_owned
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

## Second candidate: prefill_queued

Legacy `atpm_preempt_prefill_queued` is preserved as one single-nonbatch
program with two dependent rounds under the same Q2 environment. Round one
has a priority-50 placeholder and peers 30a,30b,40a,40b,30c,30d,30e,30f,70;
round two has a priority-70 placeholder and eight 70 peers plus incoming 90.
All twenty requests retain 2048/2 shape. Each round explicitly waits for the
placeholder Schedule admission and P pending >=1, submits peers with .15s
gaps, settles every Schedule, then drains placeholder before the nine peers.
Each round retains its own 30s Master clean gate before proceeding.

The observed dispatch contract remains first submitter, then descending
priority with submit-order ties. Round-one PR10 requires shape, no wave
8400/8429 and placeholder success; PR5 requires shape/no eviction; PR6
requires shape/all nine success; PR4 requires all these together. Round-two
PR10 additionally includes the placeholder in zero-eviction checks and
requires shape/all nine/placeholder success. Each P6_terminal checks shape
and both cohorts' success, conjoined with that round's Master clean.
Raw engine dispatch and both cohorts' outcomes are retained separately for
each round. This is the old EV-1-FIXED pull-model contract; it does not claim
that eviction actually happened or that BATCH victim selection was tested.
The 1200s whole-instance envelope accommodates two independently bounded
rounds; Schedule90, Generate120, each-consumer35 and clean30 are unchanged.

Complete-program tests add twenty actual consumer workers, an engine-order
inversion blocking round two, and a rejected second placeholder blocking
its nine peers. These remain external-IO fixtures, not Java execution.

## Shared cleanup dependency

Independent review found that the original base's balance_clean incorrectly
accepted Decode {inflight_requests: 0, total_load: 7}. Both candidates require
the owner fix d955a8434b9c1458e1eadb76f90734eda47b1c23 (core pick 0482),
which validates all present counters then preserves the legacy OR fallback.
The first candidate 2a664 was independently accepted only together with this
fix; bare 2a664 is not equivalent for P6. The shared owner has independent
0/7 TIMEOUT and missing-fields ERROR evidence. No local duplicate cleaner is
introduced by the preemption implementation.


## Third candidate: timeout_attribution

The old atpm_timeout_attribution A1 configuration uses queueTimeoutMs=7000
and PREFILL_QUEUED preemption, with the same 1P/4D cap1/waiting8 axes.
Round one sets P12000, waits1.5, admits placeholder90 and observes pending,
then submits eight30, incoming70 and two90. Round two follows clean30,
sets P10000 and waits1.5, then admits placeholder70 and submits eight30 plus
incoming70. All requests remain 2048/2; cohorts keep .15 issue gaps and
Schedule90/Generate120/fresh consumer35. Round-one eleven-peer drain has
385s envelope; round two retains315s.

Each PR7 requires every queued outcome to be8511, every actual response's
admission_reject_reason to be UNSPECIFIED(0), zero8400 low-peer victims and
successful placeholder completion. Each P6_terminal only checks placeholder
success and zero8400 low-peer victims, exactly preserving the old final P6
conjunction across both rounds when combined with clean30 gates. P6 is not
silently strengthened to assert8511 itself; PR7 supplies that requirement.
The incoming Schedule wall time is retained raw in the artifact with no PR8
band check, as in the old contract. Missing response/reason evidence is ERROR
rather than passing the old filtered/vacuous reason comprehension.
A failed earlier check blocks later stages; this preserves case failure but
can suppress second-round diagnostics which the old GradeReport accumulated.

Fixture tests cover all22 actual Schedule requests, two successful consumers,
twenty actual Schedule rejections and the correct-code/wrong-reason negative
where PR7 fails independently of P6_terminal. This is not Java evidence and
does not claim the old classifier gap still exists on a new Java source.


### Timeout first-round cleanup is an explicit stronger requirement

Independent review of 0d84faf found an old false-PASS path: after a passing
first-round PR7, failure of clean1 returned report.finish without recording a
failing P6, potentially reporting PASS and omitting round two. The new formal
program makes clean1 a mandatory failing check and requires both rounds for
PASS. This is a deliberate strengthening of cleanup and two-round completion,
not exact Boolean equivalence for that old early-return path. The old case
remains unchanged. Independent static acceptance of the third candidate is
conditional on this stronger construction and shared clean fix d955; it does
not constitute Java execution acceptance.


## Fourth candidate: disabled_zero_eviction

Legacy atpm_preemption_disabled_zero_eviction uses T1: 1P/4D, PRIORITY,
SINGLE/NON_BATCH, queueTimeout8000, cap1/waiting8, no preemption block.
The complete resolved configuration is tested against old _t1_spec/render_env.
To avoid deep-merge inheritance, the family no longer supplies preemption;
the first three variants explicitly add PREFILL_QUEUED instead. Their resolved
behavior is retained. Empty dictionaries and null are not used to erase it.

P3000 plus1.5 sync precedes two rounds: ph30+eight30+incoming70, then
ph70+eight70+incoming90, all2048/2. Each placeholder must be admitted/pending,
all9 peer Schedule calls settle before ph35 and eachpeer35; each round has
clean30. AT2 and P6_terminal require no8400/8429/8430 among placeholder and
eight queued peers, and incoming200 or8511. Legal expiry of queued requests
is preserved; this does not require every request to succeed. Both rounds
must finish for a PASS; early failure may skip later diagnostic work.

Unknown terminal codes produce ERROR because they cannot prove absence of a
forbidden family. General raw-trailer extraction remains a shared backend
integration dependency: a trailer-only failed stream cannot pass by being
recorded as None. No actual Java/error-family completeness is claimed until
that dependency and real execution are verified. Fixture tests cover20
successful consumers,18 legal8511 Schedule rejections, queued8430 failure,
and complete T1 config comparison. Shared clean d955 remains required.
