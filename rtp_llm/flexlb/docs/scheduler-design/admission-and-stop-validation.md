# Admission and stop validation

Date: 2026-10-26. Tests ran inside `luoli_gpu` with JDK 21 and the offline
`opensource,!internal` Maven profiles. No production settings or performance
thresholds were relaxed.

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
