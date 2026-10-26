# Admission and stop validation

Date: 2026-10-26. Tests ran inside `luoli_gpu` with JDK 21 and the offline
`opensource,!internal` Maven profiles. No production settings or performance
thresholds were relaxed.

The initial sections describe commit `e598eb5f57`. The callback-reduction
follow-up has its own source identity and results at the end of this document.

## Scope and source identity

- The before snapshot is the working tree captured before this implementation,
  including changes already pending on top of `016369ed12`.
- This implementation changes 10 production files, including the Acquirer rename
  and deletion of DeliveryCleanup: **1,350 added / 1,365 deleted, net -15 lines**.
- The commit also preserves the earlier pending source changes. Relative to Git
  HEAD, its combined production delta is **+59 lines**, not -15.
- Final source: 575 Java files. SHA-256 of `final-source.sha256`:
  `0ba3df44b2514bff3dd6cfbcbe5587b68e28d9ad2e2d6ba773d21edb98dd350c`.
- After UT, only the obsolete performance fixture was repaired. It is excluded
  from UT by its performance tag and was subsequently tested on both snapshots.
  Production source hashes stayed unchanged through those test runs. The commit
  hooks then normalized formatting/import order and replaced a test wildcard
  import with explicit imports. No production logic changed. The pre-hook
  manifest is preserved as `pre-hook-source.sha256`; budgets were checked again
  and admission/delivery regression tests rerun after hook formatting.

## Source budgets

Counts include private methods, overloads and explicit constructors, after AOSP
formatting. Nested types are counted separately and also pass the same budget.

| Class                    | Methods | Longest method |
| ------------------------ | ------: | -------------: |
| AbstractRequestScheduler |      30 |            100 |
| RequestContext           |      30 |             97 |
| QueuedRequestScheduler   |      28 |             90 |
| DecodeAdmission          |      20 |             94 |
| DeliveryDispatcher       |      20 |             97 |

## Unit tests

**3,276 tests: 0 failures, 0 errors, 1 skipped.** The Maven reactor reported
`BUILD SUCCESS`; the complete XML reports are archived in `final-ut-reports.tar`.

| Module      |           Tests |
| ----------- | --------------: |
| common      |             281 |
| cache       |              46 |
| grpc        |             208 |
| sync        |           2,131 |
| api         | 210 (1 skipped) |
| mock-engine |             400 |

The follower connection-failure fixture previously tried to reserve a recently
closed listener port and failed with BindException. It now reserves a fresh,
exclusive non-listening address; its actual connection-failure and exactly-once
recovery assertions remain in place.

## Performance

Both versions use 64g heap, ExitOnOutOfMemoryError and advanced Netty leak
detection. The 750P/750D matrix uses 10s warmup and 10s measurement at 3,000 and
10,000 QPS, minimum throughput ratio 0.98, and 50ms client/server/batch-wait P99
ceilings. Runs were sequential on a shared, unpinned host; this is not proof of a
causal speedup or absence of regression.

| Primary suite                | Before                       | After               |
| ---------------------------- | ---------------------------- | ------------------- |
| API performance, 16 tests    | 15 pass, 1 failure           | 15 pass, 1 failure  |
| Sync performance, 5 tests    | 5 pass                       | 5 pass              |
| 750P/750D BATCH, 2 tests     | 2 warmup errors              | 1 pass, 1 failure   |
| 750P/750D NON_BATCH, 2 tests | 2 pass                       | 2 pass              |
| Total, 25 distinct cases     | 22 pass, 1 failure, 2 errors | 23 pass, 2 failures |

The API burst server P99 was 501ms before and 475ms after, both above 50ms.
The initial before BATCH cases failed during warmup with `receiving budget full`.
Additional BATCH repetitions are retained separately, not substituted for these
primary results:

| 750P/750D, target 10,000 QPS | Client QPS | Client P99 | Server P99 |
| ---------------------------- | ---------: | ---------: | ---------: |
| NON_BATCH before             |    9,994.5 |   46.655ms |        9ms |
| NON_BATCH after              |    9,979.4 |   15.109ms |        2ms |
| BATCH before repeat          |    9,989.4 |   78.032ms |       20ms |
| BATCH after primary          |    9,989.3 |  149.464ms |       34ms |
| BATCH after repeat           |    9,988.4 |  103.622ms |       55ms |

The before BATCH repeat also failed its 3,000-QPS warmup. **Performance remains
unaccepted; a BATCH tail-latency regression cannot be ruled out.**

A separate 10,000-QPS GC diagnostic pair retained the same limits. The before
measurement window (JVM uptime 18.230-28.257s) contained 7 GC pauses totaling
397.663ms, maximum 70.662ms; its client P99 was 72.507ms. The after diagnostic run
failed during warmup with receiving-budget exhaustion and produced no comparable
measurement window. This establishes that long GC pauses already occur in the
baseline, but does not establish the cause of the observed before/after latency
difference. Diagnostic repetitions are not included in the 25 distinct-case total.

The stale Prefill allocation benchmark referenced a deleted private API. Both
versions now use the identical repaired fixture measuring the real selected-worker
`admissionSummary + classifyWorkerCapacity` path. Original allocation ceilings
are unchanged: measured classification is 0 bytes at 1/64/512/1024 workers;
mixed-priority summaries measure 65 bytes per query, below the 80-byte ceiling.
This does not claim to measure the removed whole-fleet aggregation path.

## Concurrent reviews

Two agents independently reviewed capacity ownership, late receipts, lock order,
exceptional cleanup and hot-path allocations. Fixes include protecting admission
handoff from observer cancellation, including executor wait in the attempt
deadline, retaining ACK work after submission rejection, and fencing empty-task
closure against an already accepted ordinary stop. Regression tests reproduce
these failures. The last independent review found no new definite ownership,
deadlock or allocation regression in the fact consolidation.

The observer-GC test failure was traced to Mockito's thread-local stubbing history
retaining the timeout Runnable. A real timing delegate replaces that mock; all
weak-reference assertions remain.

## Evidence

Local evidence directory:
`/data0/luoli.hn/data0/work/flexlb-admission-20261026`.
It contains `before.tar`, source manifests, `final-budgets.tsv`, `accepted-ut.log`,
the archived UT XML reports, individual before/after performance logs,
`results.json`, and reproducible validation scripts. Failed intermediate runs
and the obsolete-fixture failures remain separate historical artifacts.

## Callback reduction (e2e129d4db)

The follow-up change makes a Dispatcher preemption ACK with an active delivery
claim enter Scheduler once. Scheduler records both the attempt and delivery
proof under the Context identity lock, then runs the common continuation outside
that lock. Engine and delivery evidence use the same continuation. There is no
deferred-continuation flag, thread-local or new executor.

The ACK aggregation path in `DecodeAdmission` also removes an intermediate
continuation method. Ordinary replies reuse the original victim observer; only
mixed unresolved NOT_FOUND replies create a subset observer. This avoids an
extra list and per-victim minimal-stage/future copies on the ordinary path. The existing
NOT_FOUND early rejection, UNKNOWN waiting, timeout freeze and Decode-ledger
commit checks are unchanged.

New regression coverage checks both facts before the first continuation, a stale
attempt with current delivery proof, transient fact-application failure, and a
new cancellation RPC after three failed local applications. Remote proof never
completes the local settlement signal by itself.

Final UT and performance results for this follow-up are recorded separately
below; the results above describe the preceding commit.

- Production scope: three files, net +9 lines relative to `e598eb5f57`; no new
  service classes, execution threads or global locks.
- Method budgets: Scheduler 30 / longest 100 lines; Dispatcher 20 / 97;
  DecodeAdmission 19 / 97. Context and Queue are unchanged.
- SHA-256 of the 575-file `verified-source.sha256` manifest:
  `72b11cce6fce742cf846e4b39c9fa5f43234b72d11538488dcad670951ff8ac7`.
- Two independent reviews found no definite new ownership or lock-order defect.
  The Decode ledger mutation has no callbacks; capacity publication and workflow
  continuation stay outside the Context monitor.
- Protocol boundaries are retained: an ACK is not resource release, and the
  admission result remains subject to canonical victim facts and the Decode
  ledger. The admission change removes redundant continuation/future mapping;
  it does not collapse these two distinct completion meanings.

The final frozen source passed the complete UT reactor in `luoli_gpu`:
**3,280 tests, 0 failures, 0 errors, 1 skipped**, with Maven `BUILD SUCCESS`.
`verified-ut.log` contains the console results; `verified-ut-reports.tar` preserves
the Surefire reports before performance execution overwrites them.

An intermediate run's XML remains under `callback-final-ut`; its console log
was overwritten after a wrapper-edit error. That run is historical evidence,
not the final source's verification result.

The final continuation removes two extra snapshot monitor acquisitions and two
capturing Runnable evaluations per normal delivery evidence that were found in
the intermediate implementation. Independent direct try/catch blocks preserve
the remaining cleanup attempts after an effect fails. This costs source lines
but avoids introducing these extra normal-path operations merely to share code.

The follow-up performance primary set contains the same 25 cases and retains
the same heap, leak detector, workloads and SLO limits: **20 pass, 4 failures,
1 error**. Sync passes 5/5, API passes 14/16, BATCH matrix passes 0/2,
NON_BATCH passes 1/2. API failures are the burst server P99 of 562ms and the
4P/8D, 10,000-QPS server P99 of 60ms (ceiling 50ms). Matrix results:

| Mode / target QPS  | Client QPS | Client P99 | Server P99 | Result                                            |
| ------------------ | ---------: | ---------: | ---------: | ------------------------------------------------- |
| BATCH / 3,000      |        N/A |        N/A |        N/A | Error during warmup: receiving budget full        |
| BATCH / 10,000     |    9,969.5 |  261.679ms |       52ms | Fail: client P99; server P99 also exceeds ceiling |
| NON_BATCH / 3,000  |    2,995.7 |   16.785ms |        4ms | Pass                                              |
| NON_BATCH / 10,000 |    9,994.9 |   61.595ms |       10ms | Fail: client P99                                  |

**Performance remains unaccepted.** These sequential runs use a shared host and
do not establish a causal speedup or rule out regression. Final logs use the
`verified-` prefix; `verified-results.json` contains totals and matrix metrics.
The 575-file manifest still matches after all tests. Intermediate `unified-` and
`callback-final-` logs are historical only; their more favorable measurements do
not replace this final result.

Isolated burst diagnostics also leave a regression concern: parent `e598eb5f57`
measured 7,655.8 client QPS / 427.297ms client P99 / 427ms server P99, while an
intermediate candidate after eliminating the extra snapshot/wrapper costs
measured 5,804.4 / 620.921ms / 620ms. Logs are `parent-isolated-burst.log` and
`candidate-snapshot-burst.log`. These are separate diagnostic runs, not additional
passing cases or a controlled explanation of the difference. Eliminating the
identified extra operations did not establish the cause of the remaining tail
latency; this change cannot yet be certified as free of performance regression.

## Effect ownership

Review of `e2e129d4db` found that a single Scheduler entry still ran multiple
effect owners: confirmation and its caller both updated deadlines, while closing
and its caller both submitted settlement. A pre-close snapshot could also cause
cancel re-entry after the request was already ENDING. The preceding UT pass did
not establish the desired ownership contract.

The new change makes continuation branches mutually exclusive and gives
`closeRequest` responsibility for blocked/existing close recovery as well as new
terminal selection. It removes caller cleanup tails from delivery failure,
placement, retirement and deadline handling. No service class, executor, map,
Context execution state or global lock is added.

Five new action-count cases failed on the old implementation and passed after the
change (`ownership-red.log`, `ownership-green.log`). Three additional cases test
single-fact recovery of ACK deadline installation, cleanup submission, and a new
terminal deadline while sender/release proof is still absent. Together with the
combined-receipt ordering test, the final targeted set passes 9/9 in
`ownership-final-targeted.log`.

Review caught and fixed the removal of accidental recovery previously supplied
by caller duplication. CloseTask owns bounded deadline retries. `requestCleanup`
owns submission recovery, using an atomic WorkRegistration execution/ownership
check; it never mistakes an inline task's failure for a failed handoff. An
extended failure test caught that distinction before final verification.
Historical failures remain observable; successful installation is recorded
independently. The existing NOT_FOUND confirmation phase and synchronous
RETIREMENT outcome selection remain necessary boundaries.

Final source budget: Scheduler 30 methods / maximum 99 lines; ExpirationTimer
10 / 64; SchedulerRuntime 30 / 61 (WorkRegistration 10 / 29).
Production delta relative to `e2e129d4db` is three files, net +18 lines.
`ownership-source.sha256` freezes 579 Java files: the same 575 module sources plus
four existing Java tools. Its SHA-256 is
`8867fd9e7dc9814b507e8b76f92d11fc5a390bb1c86fc9d379e861c316c92f15`.

The final targeted contract set passes 15/15 (`ownership-final-contracts.log`).
It includes modeled deadline installation state, retaining the original failure
through admission completion and shutdown, and five consecutive installation
failures that must not falsely archive a request or release resources twice.
The first full run and the later failed handoff test are retained separately as
`ownership-ut-initial.log` and `ownership-contracts-before-handoff-fix.log`.

Final validation runs in `luoli_gpu` from an isolated copy of `e2e129d4db` plus
only this patch at `ownership-candidate/rtp_llm/flexlb`. Concurrent PrefillState,
WorkerBatcher and batch-test edits in the shared workspace are excluded. The
full reactor and performance results use the `ownership-` prefix independently
of earlier measurements; file isolation does not make the shared host a
controlled performance environment.

The complete UT reactor passes **3,289 tests, 0 failures, 0 errors, 1 skipped**
with Maven `BUILD SUCCESS` in `ownership-ut.log`. Surefire XML reports are
preserved in `ownership-ut-reports.tar`. An earlier isolated run was missing the
repository-level telemetry JSON fixture; its failure is retained in
`ownership-ut-fixture-missing.log`. The final run includes that fixture from the
same baseline commit, with no Java source changes.

The complete performance set retains the original 64 GiB heap, advanced Netty
leak detection, workloads and thresholds: **22/25 pass, 3 failures, 0 errors**.
Sync passes 5/5, API 15/16, BATCH matrix 0/2, NON_BATCH matrix 2/2. The API burst
fails at server P99 545ms against 50ms. The 750P/750D measurements are:

| Mode / target QPS  | Client QPS | Client P99 | Server P99 | Result                                            |
| ------------------ | ---------: | ---------: | ---------: | ------------------------------------------------- |
| BATCH / 3,000      |    2,996.7 |   66.973ms |       13ms | Fail: client P99                                  |
| BATCH / 10,000     |    9,989.3 |  277.153ms |       60ms | Fail: client P99; server P99 also exceeds ceiling |
| NON_BATCH / 3,000  |    2,998.1 |   14.104ms |        3ms | Pass                                              |
| NON_BATCH / 10,000 |    9,991.8 |   26.367ms |        4ms | Pass                                              |

**Performance remains unaccepted.** The change removes demonstrated redundant
effects and passes ownership/recovery contracts; these runs do not establish a
causal speedup or rule out performance regression. `ownership-results.json`
records totals and metrics. The complete Java manifest still matches after UT,
performance execution and commit-hook checks.

## Residual code and test cleanup

This follow-up is based on `6e23c937c7`. It removes two test-only Dispatcher
preemption overloads, the unused Scheduler claim detail parameter and a catch
that only rethrew its exception. Production code decreases by 23 lines; Java
production and test sources together decrease by 112 lines. Dispatcher has
18 methods / maximum 97 lines. Scheduler remains 30 / 99, Runtime 30 / 61,
and DecodeAdmission 19 / 96.

`DeliveryCleanupTest` becomes `DeliveryStopTest`, matching the current owner.
Its equal-valued claim case now explicitly verifies STALE rejection rather
than mislabeling that branch as a second deduplicated operation. The duplicate
ACK fixtures become one first-ACK/replay contract with both assertions retained.
Decode admission tests now verify the injected Dispatcher instead of an unused
Engine mock, configure the construction-time policy, and use a Decode endpoint
so early-exit cases cannot accidentally pass the endpoint-type guard. Priority 1
must reach endpoint selection; FIFO and DIRECT policy cases exercise the actual
frozen configuration. No failure or concurrency contract is removed.

The final source manifest contains 579 Java files. Its SHA-256 is
`df9c1ab41edb6459eef985d253e57db5330d3aaaefe4456fd3bffa76409082b4`.
The first focused run passes 171 tests; the final full run also includes the
subsequent fixture correction and combined ACK test. Final validation evidence
uses the separate `residual-` prefix.

The complete UT run passes **3,288 tests, 0 failures, 0 errors, 1 skipped**,
including all 400 mock-engine tests. The one-test decrease is the combined
first-ACK/replay contract, not removed assertions. Maven reports BUILD SUCCESS
in `residual-ut.log`; all six modules' XML reports are retained in
`residual-ut-reports.tar`. The shared workspace's obsolete
`DeliveryCleanupTest.class` was removed; historical reports remain available.

The full performance rerun passes **22/25 tests, with 3 failures and 0 errors**:
Sync 5/5, API 15/16, BATCH matrix 0/2, NON_BATCH matrix 2/2. The burst case
fails at server P99 616ms against 50ms. The 750P/750D results are:

| Mode / target QPS  | Client QPS | Client P99 | Server P99 | Result           |
| ------------------ | ---------: | ---------: | ---------: | ---------------- |
| BATCH / 3,000      |    2,996.8 |   56.561ms |       13ms | Fail: client P99 |
| BATCH / 10,000     |    9,972.5 |  160.142ms |       29ms | Fail: client P99 |
| NON_BATCH / 3,000  |    2,999.6 |   11.914ms |        2ms | Pass             |
| NON_BATCH / 10,000 |    9,991.3 |   27.457ms |        5ms | Pass             |

**Performance remains unaccepted.** These shared-host runs do not establish
a causal improvement from cleanup. `residual-results.json` records this run
independently of the earlier ownership measurements.
