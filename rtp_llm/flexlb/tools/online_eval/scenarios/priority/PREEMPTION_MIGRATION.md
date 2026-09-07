# Priority preemption migration

Owner: agent4. Source baseline: `295af797bd7ed3a842c9cad42b5722c64cd24c9a`.
This checkpoint implements fourteen complete candidate families out of fourteen old
contracts. No old case is called or deleted. Each candidate keeps separate
fixed-source review and execution evidence; this checkpoint is not blanket
Java or old/new paired acceptance.

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

## Pending candidate implementations (0)

All fourteen old contracts now have explicit candidate programs, covering
seventeen declared profile instances. Review, integration, Java execution
and old/new paired acceptance remain distinct statuses; inclusion in this
YAML does not imply those acceptance stages passed.

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


## Typed stream terminal integration

This checkpoint requires reviewed shared backend
`e8e53612d995a2f97dcdd3069a8e021c65b5df03` plus the d955 clean fix.
It adds no new legacy case count; four candidates remain implemented and ten
remain pending. Bare preemption branches without this backend cannot execute
the new typed-terminal tests successfully. Integrated verification overlays
these owned files onto the fixed e8 source, preserving the shared owner code.

All cohort drain stages now use preemption_wait, which retains the independent
Schedule joins, fresh35s per-consumer budget and actual completion witnesses.
An ordinary RuntimeError from a stream can reach the scenario verdict only
when Schedule was OK, the actual consumer completion is verified, done and
transport/exit timestamps exist, and the backend recorded a parsed integer
trailer code. Missing/ambiguous evidence and transport CANCELLED remain errors.
Actual stage and RPC deadline exceptions are not swallowed, even with metadata;
this is a deliberate strict timeout boundary rather than exact equivalence to
the old helper's untyped/typed timeout collection behavior. Core RequestBatch
wait is unchanged and does not globally permit failed RPCs.

The outcome policy preserves the legacy distinction: an in-band enum2 on an
otherwise OK stream maps to8429, whereas trailer literal2 remains2. A failed
non-CANCELLED transport uses the parsed trailer code without in-band fallback;
transport CANCELLED supplies no engine terminal even if a trailer is present.
Original transport status, raw trailer bytes and in-band fields remain separate.
The disabled case's literal-trailer2 counterexample passes its old exclusion
predicate; it is not being mislabeled as a successful business request.

Six additional tests include five complete programs with actual consumer
workers and real protobuf ErrorDetailsPB bytes: typed8430 reaches the disabled verdict and
fails it, literal trailer2 retains2, CANCELLED is not an engine victim, missing
metadata is ERROR, and DEADLINE_EXCEEDED is TIMEOUT; an in-band policy test
retains2-to8429 mapping. These are local external-IO fixtures, not Java runs or
acceptance of the ten outstanding victim/error-family contracts.


## Fifth candidate: comparator_frozen_weak

The old atpm_comparator_frozen_weak runs two independently constructed
environments: Q2 priority with PREFILL_QUEUED preemption and queue60000, then
F1 FIFO without preemption and with queueTimeoutMs omitted. Both keep 1P/4D,
SINGLE/NON_BATCH, cap1/waiting8 and default perf. The full resolved configs
are compared to the old _q2_spec and _f1_spec, not just selected fields.

Each half sets P3000 and waits1.5, admits ph30 and observes P pending within6s,
then submits five2048/2 peers [30,30,70,70,70] with .15s gaps. Schedule90,
Generate120, ph35, five fresh35 drains and clean30 are preserved. Priority
dispatch is submit indices [0,2,3,4,1]; FIFO is [0,1,2,3,4], using actual
Prefill running_ms with settlement-rank/ID ties. All twelve consumers and
both placeholders must succeed. PR9/P6 conjoin both half verdicts and the
preceding mandatory per-half clean gates. Half predicates are collected as
scalar evidence; PR9/P6 fail at the final aggregate after both halves run.

Shared environment_reconfigure from fixed
7a1d3aa2f91ba1a3cd46c7b18c4b792ff99608c3 performs complete override replacement
and owned cleanup before the F1 setup. Its new epoch invalidates live handles;
the program reacquires the fleet and retains only historical scalar verdicts.
No runtime hot reload or comparator mutation is claimed: this is the old weak
construction-time contrast. The environment transition has a 300s stage
envelope; business polling and consumer budgets are unchanged.

Complete-program fixtures cover both configs, twelve consumer completions,
FIFO inversion failing the second half, and rejected first placeholder
preventing the second environment. Fixed7a1 plus these owned files is the
verification composition; this branch alone lacks the shared environment
registration/backend. These tests are not Java execution acceptance.


## Sixth candidate: config_strict_reject

Legacy atpm_config_strict_reject sends three raw JSON payloads directly to
the Java Master: removed top-level autoTpmEnabled, FIFO with defaultPriority,
and DECODE_ENGINE_OWNED without engineCancellation. Each legal base uses
1P/4D, SINGLE/NON_BATCH, cap1/waiting8 and omitted queueTimeoutMs; payloads
are fully compared with the old _prio_config/render_env plus identical raw
mutations. The third legal base retains both DECODE_RESERVED and
DECODE_ENGINE_OWNED with ack50/completion1000 before removing cancellation.

The program runs all three shared environment_startup_probe stages before
AT1, so a normal unexpected success or unrelated startup error still permits
the remaining diagnostic probes. AT1 requires all three rejected outputs;
P6 uses the third probe's current_absent_before_cleanup, matching the old
final EnvManager.current test. Successful forced teardown cannot turn P6
green. Ordinary timeout/IO exceptions remain TIMEOUT/ERROR and stop dependent
work rather than masquerading as strict parser rejection.

The shared7a1 interface requires an owned exited Master and this epoch's
private parser log. This is stronger evidence than merely finding parser
text in any exception. It also adds a legal initial setup before the three
raw epochs, an explicit framework setup requirement absent from the old
case. Each probe has a300s outer setup/cleanup envelope; the unchanged
harness startup readiness/liveness loop retains its90s cap with early exit
when the owned process dies. No Java timeout was increased. Final cleanup
is mandatory, including partial starts, and historical probe artifacts remain.

Four full-program fixtures cover exact raw payloads and cleanup ordering,
unexpected third success failing AT1/P6 despite cleanup success, unrelated
first error failing AT1 while all three probes run, and actual deadline
remaining TIMEOUT. All process observations in these tests are external-IO
fixtures; actual Java strict-parser acceptance remains pending.


## Seventh candidate: decode_engine_owned

Legacy atpm_preempt_decode_engine_owned now asserts its executable EV2
baseline, rather than the earlier descriptive claim that decode victims
are evicted. D1 keeps2P/4D, cap3/wait8, queue60000, PRIORITY/SINGLE/NON_BATCH,
DECODE_RESERVED+DECODE_ENGINE_OWNED and cancellation ack50/completion1000.
Its FLEXLB_MONITOR_METRIC_WHITELIST=flexlb_auto_tpm must survive the typed
environment metric_whitelist interface from fixed shared commit
d9da0cfde685e72b30224f204431689ce319403f. Both full resolved_config and
make_env_spec.master_env are compared against the old D1 spec.

P4000 and sync1.5 precede both waves. Each submits four priority30 occupants
with2048/500 and .15s gaps, settles their Schedule90 calls, and only then
injects the old fixed6291456 KV pressure into all four Decode owners.
After sync1.5, an explicit snapshot guard checks all Decode available KV
are zero before incoming70/2048/2 is submitted. Between waves pressure is
cleared and another1.5s sync permits the second occupant cohort to route.
Wave two additionally polls each occupant, sequentially with a fresh20s
budget and .1s interval, until actual Decode lifecycle says RUNNING.
Generate120 and fresh35s per-consumer drain remain unchanged.

Reserved PR6 requires no8400/8429 occupant terminals, all four success and
incoming rejected in{8403,8402,8510,8431}. Owned PR6 requires no8429, all
non8429 survivors successful and the same incoming reject family. PR10
conjoins the two original zero-eviction predicates. P6 preserves owned
incoming rejection and survivor success, with the preceding clean30 gate.
Both rounds retain clean30. A missing/malformed available-KV counter is
ERROR rather than the old helper's -1 sentinel, which could falsely prove
saturation. Pressure cleanup is registered before injection and uses the
owned environment's captured ops even on partial failure.

The old first clean failure can return report.finish after passing reserved
PR6 and omit round two; the new mandatory clean gate deliberately closes
that false-PASS path. Failure of a PR6 check stops later diagnostic work.
In particular, the old conditional verify_engine_cancelled and AT5 closure
computation can run only on an already failing EV2 owned-victim path. The
new fail-fast program does not claim that conditional AT5 band, live Cancel
ACK or actual eviction coverage; it retains raw failed terminal evidence.
AT5 has no observation object on any passing old EV2 execution. The live
victim contracts remain separate pending migrations.


Seven focused fixtures cover complete ten-request execution with eight
verified consumers and two actual Schedule rejections, the exact D1 config
and startup env, successful incoming failing EV2, one available Decode
blocking incoming, missing capacity evidence as ERROR, independent20s
per-request RUNNING windows, parent deadline capping, and malformed
RUNNING timestamps as ERROR. These are external-IO fixtures over actual
consumer workers, not Java acceptance.


### Comparator runtime finding and control-flow correction

The frozen0be83d8363 Java comparator run failed its original priority
order predicate: expected[20002,20004,20005,20006,20003], observed
[20004,20005,20006,20002,20003]. Placeholder and all five wave consumers
succeeded. Raw evidence is under
/tmp/agent1-yaml-evidence/preemption-env0be83/comparator; the raw
preemption-comparator-6c066f8a256e4d6cbe27b0db9de9d34f.json SHA256 is
d1f476c24c290ee49a4b58f31bdfe2dc89d57fef76f1e155997187d72071e59f.
That result remains FAIL; it does not establish a root cause or a valid
replacement ordering contract.

The initial formal half check stopped before clean1/FIFO setup, so the
second environment was not exercised. The old run_half returns a Boolean
tuple on an order failure and continues the FIFO half, then fails PR9/P6
at the aggregate. The corrected adapter records half passed/shape/all_ok
and raw actual/expected order as evidence, without a premature check. The
unchanged final PR9/P6 conjunction reports FAIL after both halves. Existing
placeholder and cleanup gates remain mandatory and can still stop the
program; in particular, stopping on failed clean remains stricter than the
old diagnostic continuation after a dirty half. No ordering predicate,
request shape, configuration, timeout or acceptance threshold is changed.

A complete-program regression reproduces the observed pure-priority first
half, verifies12 consumers/two environments/FIFO clean, and asserts final
PR9/P6 FAIL with priority_half=false and fifo_half=true. The existing FIFO
inversion still yields final FAIL; construction failure still prevents
reconfiguration. This correction provides missing diagnostic coverage,
not a green result for the original runtime failure.


## Eighth candidate: error_code_family

The old atpm_error_code_family contains three executable NON_BATCH segments;
the reserved BATCH8514/8515 arm remains unimplemented and is not claimed.
Segment1 uses C1 2P/2D, global outstanding2, queue/inflight/waiting omitted,
and no preemption. Segments2/3 rebuild Q2/A1 1P/4D, cap1/wait8, PQ preemption,
queue60000/7000. All three complete resolved configurations are compared to
old _c1_spec/_q2_spec/_a1_spec. The cdd856f699fddbfd16c1a9456ee5a15bbf1b8be0
shared interface preserves actual initial4 workers and pre-reserves max5;
no C1 topology is silently changed to avoid the transition.

Segment1 sets both P4000 and waits1.5, admits two50 placeholders with .15
gaps and observes pending on the first actual P. Two arrivals30/70 must
both terminate8502, with actual settled-minus-submitted strictly below3s.
Both placeholders must succeed; forbidden families are8402/8403/8431/8400/
8429/8511. Rejection reasons are retained as diagnostics only, exactly as
in the old executable predicate (the old detail says expected0 but does
not enforce it). After restoring both P100, a single recovery uses unset
priority,2048/2, unique block key RID*100+1, Schedule30, Generate60 and
observation30. Its actual consumer success is required, followed by clean30.
The shared prepared recovery driver may allocate an unused wire ID; no
contiguous-ID assumption is part of this program's contract.

Segment2 sets P3000/sync1.5, then ph70 and eight70+incoming90. All ten
requests succeed, with peer dispatch indices[0,8,1,2,3,4,5,6,7] from actual
Prefill running_ms/settlement-rank/ID. The no8502/8403/8431/8400/8429/8511
isolation and clean30 remain. Segment3 sets P10000/sync1.5, then ph70 and
eight30+incoming90. All nine queued terminals must be8511, low peers have
no8400, placeholder succeeds, and the original isolation/clean30 remain.
Incoming rejection reason remains diagnostic here too. Each segment retains
Schedule90/Generate120, fresh35 consumer drains, .15 peer gaps, ph pending
6/.1, and a separately bounded environment rebuild before the next segment.

Segment results are scalar evidence until the final AT4/P6 conjunction,
so ordinary wrong-code/slow-rejection/shape failures do not hide later
segments. Missing/ambiguous evidence and failed clean gates stop execution
and trigger owned cleanup; the latter is an explicitly stronger stopping
rule than the old report's continuation after dirty segment cleanup.
Recovery is an owned RequestBatch-based worker with completion witnesses,
not a call to the legacy case or an assumed success.


Four complete-program fixtures cover25 Schedule requests,14 actual consumer
completions, all three exact configs/topologies and recovery key/priority;
a wrong segment1 code still runs segments2/3 but fails final AT4/P6; an
actual3.1s Schedule rejection fails the unchanged3s bound; reason99 remains
diagnostic and does not add an old-absent predicate. The fixed cdd source
plus owned preemption files is the verification composition, not bare
branch execution or Java acceptance.


## Ninth candidate: decode_reservation_priority

The old atpm_decode_reservation_priority reuses the exact D1 environment
from decode_engine_owned, including its metric whitelist. Its executable
EV2 form contains three waves, each with four occupants and one incoming.
P4000/sync1.5, .15 occupant submission gaps, Schedule90/Generate120 and
fresh35 consumer drains are retained. Each wave waits for all four actual
Decode RUNNING lifecycles, sequential fresh20/.1 with an80s parent cap,
before injecting6291456 KV pressure into all four Decode owners and
waiting1.5. Between waves pressure is cleared and sync1.5 permits routing.
Unlike decode_engine_owned, this old case has no explicit post-injection
saturation guard; that extra predicate is not copied into this variant.

Wave1 uses four30/2048/500 and incoming70/2048/2. It requires zero8429
victims, successful survivors, and incoming success or the old EV2 reject
family{8403,8402,8510,8431}. The victim metric delta is exactly zero for
labels victim_priority30/incoming_priority70. Wave2 uses four50 and
incoming50, all other lengths unchanged. It requires zero8400/8429, all
occupants successful, incoming code in{200,8511,8403,8402,8510,8431}, and
an unfiltered victim metric delta of zero. Wave3 uses two30/2048/500 and
two30/16384/500, incoming70/8192/2; zero8400/8429, successful survivors,
and incoming success or EV2 reject family. It has no metric predicate,
and8511 is not silently added to its incoming legal set. No kvBucket
victim preference is claimed observable on this zero-victim EV2 baseline.

Metrics are scraped before injection and after consumer drain for waves1/2
only, from one fixed management endpoint per environment. The old
name-substring and label-subset selection is retained. An absent matching
series in valid Prometheus output remains value=None, separately marked
missing_series, with the old (after or0)-(before or0) arithmetic. This is
explicitly weak evidence and does not prove that a zero-valued series was
exposed. Failed HTTP/invalid exposition/malformed matching samples are ERROR
instead of becoming old empty-list/zero fallbacks. Raw response attempts
and selected samples remain in artifacts; finite nonnegative counters are
required. No home/shared log path is read by this metric observation.

Wave predicates are scalar evidence until final AT7/P6 aggregate; ordinary
terminal or metric-delta failures still allow later waves to run. Each
clean30 remains mandatory; stopping after failed cleanup is stronger than
the old diagnostic continuation (the old AT7 could still be reported
separately when only clean3 failed). Owned pressure cleanup remains active
on exceptions and final teardown. This is not live eviction/Java acceptance.


Seven focused fixtures cover15 Schedule requests and12 actual completed
consumer workers, exact D1 configuration/startup env and mixed-size wave3,
metric delta causing final failure after all waves, explicit sparse-series
weak-zero arithmetic, unavailable scrape failing before incoming,8511
rejected in wave3 while legal in wave2, fixed fallback endpoint/label scope,
and malformed/nonfinite victim samples as ERROR. Full verification uses
fixed cdd856 plus these owned files. No actual Java result is implied.

Independent review follow-up: validate complete classic-name victim sample
label blocks before applying the label subset. Unquoted values, duplicate
label keys, missing separators, trailing garbage, and duplicate series
raise ERROR; they cannot disappear into the sparse-series zero fallback.
The valid absent-series arithmetic and all wave thresholds are unchanged.
Two additional regressions exercise malformed labels through the real
metric action and through the complete program: baseline ERROR blocks
incoming and later waves while retaining the raw metric error artifact.


## Observability integrity (O1)

`observability_integrity` ports the executable `atpm_observability_integrity`
contract on fixed shared 4e2139. Exact O1 is 1P/4D, SINGLE/NON_BATCH,
PRIORITY, inflight1/waiting8, PREFILL_QUEUED, queueTimeout7000,
metric_whitelist=flexlb_auto_tpm and master_debug_log=true. The latter
controls the old single-Master DEBUG loggers; it is not debug_enabled or
FLEXLB_DEBUG_ENABLED. Actual EnvSpec perf/topology/config/master_env and
logging flag are compared to `_o1_spec` in the fixture.

Prefill3000/sync1.5, admitted placeholder50/2048/2 and pending6/.1 precede
a Schedule-only duplicate of the exact placeholder RID: priority40,
2048/2, RPC30. Code8406 is required only by AT6. No consumer opens for
that probe; an RPC exception remains recorded and allows the wave to
continue as before, while stage expiry still propagates. The nine wave
priorities are30/30/50/50/70/70/30/30/90, gap.15, all2048/2. Schedule90,
Generate120 and fresh per-request drain35 (wave315) stay unchanged.

Client shape remains exactly ph+30a+90 successful, seven8511, no
8402/8510, and actual Prefill running_ms/Schedule-settle-rank/RID order
30a before90. Only the successful pair requires lifecycle dispatch
records; absent or malformed pair evidence now raises ERROR rather than
using the old missing-timestamp sentinel. Each original request keeps
Schedule and actual consumer terminal evidence separately.

Request counter buckets remain30:4/50:3/70:2/90:1. A schedule success
metric need only be present (zero is legal). Victim total remains the
executable old `(total or0)==0`; the old docstring's value1 is stale and
is not the executable assertion. Missing victim series stays explicit
weak-zero evidence. Metric source validation uses the ninth candidate's
fixed endpoint/strict victim sample checks; required request/schedule
metrics retain the old substring/subset parser semantics.

Read only this single Master's stdout and private master-logs/flexlb.log
and pv.log under instance artifacts; no home-directory fallback. The
scheduler marker may appear in either owned Master log. Parse prefixed
PV JSON records, filter exact integer requestId to the ten owned RIDs,
then retain the last400 matching records. An admissionRejectReason key
with null value still proves channel availability. Exact RID filtering
closes the old substring-prefix ambiguity. Missing files make the plane
false; path ownership errors or reads above16MiB raise ERROR. Raw owned
log text and selected records are retained in the observation artifact.

Metrics/log/PV failures are scalar until final AT8, so P6 and AT6 retain
independent verdicts. After mandatory clean30: AT8=client+planes,
P6=client+clean, AT6=duplicate8406+client+clean. As with the other migrated
programs, cleanup failure hard-stops instead of the old report's soft
continuation; it cannot produce PASS. RestoreP100 and owned teardown
remain registered. Six complete-program fixtures cover positive evidence,
missing required metric, duplicate accepted without an extra consumer,
foreign RID prefix, reversed actual dispatch and missing private DEBUG
marker. No Java or paired acceptance is claimed.


## Live Prefill queue replacement

`prefill_queued_live_single` and `prefill_queued_live_window` implement
one old family, `atpm_preempt_prefill_queued_live`, on its two declared
profiles: single-batch and batch-window. Existing ten families remain
single-nonbatch; the compiled checkpoint has12 instances/11 families.
The two actual EnvSpecs match `_pq_live_spec`: 1P/4D default perf,
PRIORITY+BATCH, waiting2, queue60000, PREFILL_QUEUED only, auto_tpm
whitelist. SINGLE stays SINGLE; FIXED_WINDOW retains32/400ms and the
profile's other limits. The NON_BATCH request inflight cap is omitted,
while the original BATCH quota remains intact.

P4000 and sync1.5 precede a deferred Schedule-only placeholder50/2048/2
and the original pending6/.1 precondition. Three owned Schedule workers
issue victim_a30, victim_b30, incoming70 (all2048/2); exactly .15s between
submissions, without an extra sleep after incoming. Schedule RPCs stay90
and settle before any FetchResponse. The owned live-start action reuses
RequestBatch and PriorityWave's worker/cleanup accounting while preserving
this between-submissions-only timing.

Consume only placeholder, victim_a, incoming, sequentially, with separate
60s Fetch transport and fresh45s observation clocks, matching old
start_stream/wait_end. A Schedule-rejected survivor is skipped as before;
victim_b never gets a consumer from this choreography. Each successful
survivor requires real Fetch output, transport terminal, consumer exit and
completion verification. Request threads and partially issued batches
are registered for cleanup before dispatch.

PR10 retains exact victim_b Schedule8400 with success=false, victim_a and
incoming Schedule200+success, and all three survivors complete. PR5 is
victim_b8400 plus no victim lifecycle entry in the engine snapshot. The
snapshot must contain the full declared role/name inventory and typed
lifecycle maps; absence cannot prove never-delivered. PR6 retains stage
prefill_queued victim_count==1 and priority_preempt_count>=1, plus actual
victim8400. The 30<-70 tagged counter is diagnostic only in the old
executable predicate and remains so. An8429 victim is not accepted.

Predicates stay scalar until the final report, allowing independent P6:
Master clean30, engine inflight0/leak=false within30 (poll.5), then one
fresh recovery request (unset priority,2048/2, unique RID*100+1 key,
Schedule30/stream transport60/observation30). Recovery runs with Prefill
still at4000, as in the old source. RestoreP100 follows the final report,
with registered cleanup also covering failure paths. Missing engine
owners/typed fields, failed metric HTTP, invalid victim samples or consumer-observation
expiry raise ERROR/TIMEOUT instead of weak old defaults/soft diagnostics;
The priority_preempt series still uses the old permissive parser.
Master cleanup remains a mandatory hard gate, an explicit stronger
boundary. Ordinary PR failures still allow engine clean and recovery.

Focused complete programs cover both profiles' exact configuration,
zero Fetch until all four Schedule responses, serial Fetch IDs1/2/4,
three completed consumers plus recovery, exact8400 versus8429, a victim
seen by an engine, missing required victim counter, diagnostic priority
tags and missing engine inventory. Existing ten compiled plans retain
identical environment/stages/execution/resource_budget values after
adding the two profiles. This is fixture evidence, not live Java eviction
or paired acceptance.


## Live Decode reservation replacement

`decode_reserved_live_single` / `decode_reserved_live_window` port
`atpm_preempt_decode_reserved_live` on the same SB/BW profile pair.
The checkpoint now contains12 families/14 compiled instances. Both
actual EnvSpecs are compared to `_dr_live_spec`, including the four-block
Decode cache pool: 1P/1D, default perf, PRIORITY+BATCH, waiting8,
queue60000, PREFILL_QUEUED+DECODE_RESERVED without engineCancellation,
SINGLE or original FIXED_WINDOW32/400, auto_tpm whitelist.

Reuse the explicitly owned live Schedule/deferred Fetch machinery from
PQ: P4000/sync1.5, placeholder90/512/2 admitted and pending6/.1, victim30/
512/2 then incoming70/3500/2 with only one .15s inter-submission gap.
All three Schedule RPCs (90s) settle before serial survivor Fetch of
placeholder then incoming. Each has transport60 and fresh observation45;
victim receives no Fetch. Original survivor completion must be backed by
actual output, terminal transport and verified consumer exit.

PR10 requires victim Schedule8400 with success=false (never8429),
incoming Schedule200+success and both survivors completed. PR5 is the
exact8400 plus no victim lifecycle entry in the complete engine fleet.
PR6 is actual8400, stage=decode_reserved victim_count==1 and
victim_kv_tokens>=428. No production acceptance or engine-owned cancellation
is inferred from a master-local shadow eviction. The 428 threshold and
3500-token incoming are unchanged; a427 sample fails only PR6.

Masterclean30, engineclean30/.5 and one independent fresh recovery
request form P6, still at P4000 before restore100. Hard cleanup and strict
owner/victim-sample evidence boundaries are inherited and explicitly
stronger than the old diagnostic defaults. Four additional complete-
program tests cover both actual profile/config/cache-pool constructions,
zero Fetch until all three Schedule responses, two serial real survivors
plus recovery,427 versus428,8429 versus8400 and engine-seen victim.
These and the preceding PQ fixtures are run together; no Java or paired
result is claimed.


## Cancel NOT_FOUND construction

`cancel_not_found` ports `atpm_preempt_cancel_not_found` on its only
single-nonbatch profile: exact `_nf_spec` 1P/1D default perf, PRIORITY,
SINGLE/NON_BATCH, queue60000, inflight1/waiting8, decode maxEngineRequests1,
all three victim stages with ack50/completion1000. Actual EnvSpec config,
perf/topology/master_env are compared to the old factory. This raises the
checkpoint to13 families/15 instances; tombstoned remains pending.

An admitted victim30/512/200 reaches actual Decode RUNNING (10s/.1),
then pause.6, capture engine Cancel RPC baseline, and inject the old
status_no_respond type into the sole Decode owner. Shared status_control
binds cleanup to that exact environment/owner before the POST; explicit
clear and idempotent teardown clear both keep the original type format.
The frozen window remains3s/.05 for engine-side non-running end_state;
no health-demotion window is widened. Incoming70/512/2 follows only after
that evidence. Both Schedule RPCs remain90, direct stream transport120,
and the victim has its original fresh35 drain. An unexpectedly admitted
incoming keeps its owned consumer but is not mistaken for the required
8431 rejection and gets no extra business wait stage.

PR6 requires incoming Schedule8431 with success=false, full normal victim
output with actual consumer completion, and no victim entry in engine
cancelled_rids nor a lifecycle end_state=cancelled. A finished construction
probe alone does not prove normal output; the later two facts are separate.
AT5 independently requires aggregate per-engine Cancel delta>=1. Missing
cancel keys inside valid per-engine RPC maps retain old zero semantics
and are explicitly listed; missing maps/invalid counts are ERROR with raw
error artifacts. Before/after snapshots must share the same environment
and engine-name inventory. These black-box predicates do not expose or
assert a literal Cancel ACK enum.

After the census/verdict, clear the fault, Masterclean30, engineclean20/.5,
and one fresh unset-priority2048/2 recovery (Schedule30, transport60,
observation30, unique RID*100+1). P6 is independent of PR6 and AT5 after
the mandatory Master cleanup gate. No performance override is added.
Missing lifecycle/owner/cancellation fields and timed-out construction
raise ERROR/TIMEOUT rather than old soft/default evidence; a three-second
construction miss blocks incoming and still clears the injected fault.

Six complete-program regressions cover exact original environment and
payloads,8431 versus8403, zero Cancel delta, normal output accompanied by
engine cancellation evidence, the unchanged three-second finish window
and cleanup, and missing census map failing before injection. No Java,
ACK-enum or paired acceptance is claimed.


## Cancel TOMBSTONED construction

`cancel_tombstoned` is the final old family,
`atpm_preempt_cancel_tombstoned`. Its enqueue_batch requirement selects SB
and BW. Both actual EnvSpecs match `_ts_spec`: 1P/1D default perf,
PRIORITY+BATCH with SINGLE decision even on the BW profile, inflight
request cap omitted, waiting8, queue60000, decode maxEngineRequests1,
all three victim stages with ack50/completion1000. This final checkpoint
contains14 families/17 profile instances, not17 independent old cases.

Victim30/512/5000 is admitted with Schedule90, then opens Fetch transport60
and must produce its first output within15s (poll.02). Only then inject
crash_after n1 into original prefill-0. Register an epoch/owner-bound
restart guard before issuing the sacrificial Schedule: unset priority,
2048/10, RPC8. The output10 is the old EngineOps DEFAULT_OUTPUT_LEN,
not the output2 used by explicit recovery calls. The sacrificial response
or RPC error is diagnostic, never the crash proof, and it gets no Fetch.

Preserve the two30s/.5 Master health observations (dropped<=0, restored>=1),
restart the same named/addressed Prefill through typed engine_control,
then reconnect3. Victim cut observation10 requires an actual exited
consumer with business_finished=false; a local cancellation cannot supply
that proof. A separate immutable cut artifact preserves its pre-cleanup
record. Original transport UNAVAILABLE is not recast as engine8429.
The restart guard runs before fault cleanup if construction stops with
the engine down. Stage/invalid-owner failures hard-stop construction and
still execute cleanup, a stronger boundary than the old best-effort path.

After restart/cut, capture fresh aggregate Cancel census, issue incoming
70/512/2 (Schedule90), then serial Fetch transport60/observation45. Poll
Cancel delta for15s/.5 and retain the separate final sample the old case
performs after wait_for: reached=true and final delta>=1 are different
facts. The direct late Enqueue goes once to the victim's original Prefill
route, with the original RID, batch_id=RID*10+1, dp_rank0 and attach30000.
Its GenerateInput retains the old default input shape (2048) with explicit
output2. RPC10 must return zero successes and exactly one error for that
RID with typed code8429. Probe RPC errors remain a false fence predicate,
allowing the other planes to run; no literal Cancel ACK enum is claimed.

Engine drain remains60/.5. The Master ledger intentionally need not be
zero: poll scheduler_inflight for20s/1 until<=1, then sample again after8s
and require no growth. Missing/non-numeric owner evidence is ERROR,
never default zero. A fresh unset-priority2048/2 recovery with unique
RID*100+1 and Schedule30/transport60/observation30 completes P6.

Final PR10=cut+incoming success+final Cancel delta>=1;
PR6=exact8429 fence+Cancel reached; P6=engine clean+bounded non-growing
residue+recovery. These remain independent. There is no added Master-zero
gate and no requirement that a crash-cut victim finish business normally.
Six focused tests (two profile positives, seven complete programs) cover
original config/payload/defaults, real crash-ended victim and survivor
consumers, wrong fence code,1->2 ledger growth, normally completed victim,
reached poll versus later zero delta, and sacrificial RPC failure followed
by independently observed crash/restart. Existing fifteen compiled plans
retain identical environment/stages/execution/resource_budget values.
Fixture crash behavior does not establish live Java memory retirement,
real TOMBSTONED ACKs, or old/new paired acceptance.


## Frozen Java evidence remains separate

The independently audited `a22f0678a2beb479c3da7ff9fa09df9c354f3d19`
two-case run contains one PASS and one FAIL. `config_strict_reject` passed
all three actual Master parser-rejection probes after the raw-config
Master-only fix. Comparator retained priority_half=false/fifo_half=true,
so PR9 and composite P6 failed even though all12 consumers completed
normally. The older `0be83d8363b6bec01b0d998d3f1c57d710b35b48`
FAIL+ERROR evidence is preserved separately. These runs neither cover
all14 families nor constitute an old/new paired acceptance result.

Final local checkpoint verification: fixed shared
`4e2139c1398f1ed3039d235d2f736456612fa0f8` plus the owned preemption
files passed all78 preemption unittest cases in321.827s. The TOMBSTONED
focused six tests independently within that local work passed in80.179s.
These are actual Python driver/consumer fixture runs, not Java runs.
