# Added-worker traffic and profile correction

Reference: `a22f0678a2beb479c3da7ff9fa09df9c354f3d19`,
`elastic_stop_after_add` and `harness._pump_until_accepted`.
This corrects the BW candidate and supplies explicit SB/SN/WN variants together.
It does not relabel earlier candidate results as acceptance of this revision.

Two legacy predicates differed in the earlier candidate:

* The 15s pre-stop and 20s post-start windows bound request issuance. The legacy
  serial loop waits for its last issued request, then checks accepted counters,
  even if completion crossed the window. The previous independent ColdFlow plus
  `elastic_accepted` required counter growth within the window itself. The new
  dedicated probe restores serial issuance, completion, counter check, and a
  200ms pause after an unsuccessful attempt. It takes a final sample on expiry.
* Legacy requires cumulative `accepted_after > accepted_before_stop`, as well as
  fresh post-start growth. The former was missing. A separate cross-restart
  comparison now rejects a reset counter that received one fresh request but
  remains below the pre-stop count. Each pump also compares against its own
  initial counter, so existing traffic cannot establish fresh traffic.

Schedule retains 30s, stream 10s, with a 40s request budget. Dedicated stage
budgets of 60s/65s cover the 15s/20s issuance cutoff plus the last request and
counter reads. Missing counters are errors. Probe errors remain recorded and do
not introduce a new success-rate threshold: these probes establish accepted
traffic; the separate survivor request must complete without business errors.

All profiles preserve full old resolved configuration, fault performance,
private 2P/4D topology, dynamic discovery, both cache pools, PRIORITY ordering,
and omitted queue timeout. A single Prefill is added through the existing
control action; stop/start operates on that same added engine. Only this family
requires `queue` instead of `enqueue_batch`. No batch-zero assertion is used to
claim route-owner cleanup. Birth configuration is inherited from the same mock
environment; no added-worker override is introduced.

Explicit strengthening remains: typed counter evidence, consumer completeness,
and expected-protocol checks on both pumps. Fetch follows the actual Schedule
response and does not establish that all requests reached EnqueueBatch.
NON_BATCH reconstructs the same Generate shape and copies role addresses.
The prior candidate's exact discovered/alive topology gates and early failure
blocking remain stronger than legacy's final conjunction and stopped `alive<=2`
condition. The survivor uses one isolated key `[7]`, the candidate remapping of
legacy's case-base-relative `[base+7]`; pump keys remain `[RID*100+1]`.

Tests execute actual RecordedRequests with simulated Schedule/Fetch/Generate
responses, verify full configuration equality for four profiles, and retain
terminal/consumer records for pumps and survivor. Regressions cover wrong
protocol with successful streams, counter reset, stale baseline, no resumed
traffic, survivor failure, and a 21s Schedule finishing beyond either issuance
window but within its 30s timeout. These are Python construction tests, not Java
acceptance. Independent static and fixture review passed for
`8ed827d732017d5de16fd52be65189b394e61f08`: the reviewer independently ran eight
added-worker tests in 0.458s and 99 elastic tests in 20.748s against core4e,
and verified all four 17-stage plans with one dynamic addition. This signature
depends on the integration owner completing the catalog registration below and
validating the default entry point. Legacy cases remain available; real Java
acceptance is still pending.

Integration requires two catalog entries (owned branch lacks the shared catalog):
import `HANDLERS as ELASTIC_ADDED_WORKER_HANDLERS` from
`.actions.elastic_added_worker`, then include `*ELASTIC_ADDED_WORKER_HANDLERS` in
the descriptor list. Tests explicitly supply these handlers; the integration
owner applies the catalog change. No shared `elastic.py` edit is required.
