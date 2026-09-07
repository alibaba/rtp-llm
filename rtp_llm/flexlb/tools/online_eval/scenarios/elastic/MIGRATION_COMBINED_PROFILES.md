# Combined lifecycle profile migration

Reference: `a22f0678a2beb479c3da7ff9fa09df9c354f3d19`, the four legacy IDs
`elastic_add_flow`, `elastic_remove_flow`, `elastic_add_remove_cycle`, and
`elastic_add_preference`. The existing combined normal/strict construction is
retained, with BW corrections and six explicit new variants for SB/SN/WN at
both grades. These cover 12 additional unique legacy-ID/profile pairs, not 24.
Each program has 57 stages and four dynamic Prefill additions in private 2P/4D.

## Request cohorts and corrections

The combined add/preference construction keeps its declared 15s pre-add baseline,
45s post-convergence window (10s transient plus five 7s steady subwindows),
steady newcomer ceiling <=0.6 normal / <=0.5 strict, and old-worker floor >=0.1.
The transient slices remain observations. The separate add availability cohort
is issued from add-start minus 1s through first-accepted observation plus 1s;
its >=90% check prevents the longer preference baseline from diluting failures.
This remains an explicit combination of two legacy cases rather than an exact
replay of the standalone add case's entire flow lifetime.

Remove now starts its background flow and then runs a separate serial accepted
probe with a 10s issuance window. The prior candidate only watched counters of
that background flow and added a 0.5s pause; the extra pause is removed. Each of
three cycles now runs its separate 15s issuance probe before starting that
round's background flow, then waits the original 0.5s before graceful removal.
The prior candidate started the flow before probing, mixing probe-time traffic
into the zero-error cohort. The serial probes preserve Schedule30/stream10,
one cold key `[RID*100+1]`, completion-then-counter-check, 200ms pause after an
unsuccessful attempt, and a final counter read after the issuance window.
A last issued request may finish beyond the 10s/15s cutoff.

The remove background cohort runs through the removal and original 3s hold;
each cycle's separate background cohort stops immediately after removal returns.
All five flow cohorts are retained separately. The remove/cycle zero-error gates
remain 100%, while preference remains >=90%. Serial probe outcomes are recorded
but have no new business-success threshold: a probe's accepted-counter purpose
must not be confused with the background cohort's zero-error contract.
All profile protocol checks are separately named construction checks.

## Literal nonbatch accounting

NB variants use a dedicated literal-accounting action. The old predicate reads
Prefill `inflight_batches`, defaulting an absent key to zero. That behavior is
preserved; scheduler and Decode fields retain their own checks. The raw
`inflight_route_requests` field, when present, stays in evidence and is not a
hard gate. A fixture explicitly permits literal PASS with route count 7 and an
absent batch key. This PASS cannot establish that the Prefill route owner has
released requests. No additional route-owner assertion is folded into old IDs.
Existing stricter typed scheduler/Decode evidence and nonempty endpoint lists
remain declared construction guards. Other accounting callers retain their
original strict missing-Prefill-field behavior.

## Cycle recovery observation and cleanup

The final request restores the legacy cold key `[RID*100+1]`, Schedule30 and
Fetch/Generate RPC60. It reuses RecordedRequests.run; there is no second reader.
A specialized _activate hook records real open-return only after the stream
RPC returned its iterator. The caller then records its own observation-start
timestamp and waits up to 30s. It never derives this start from the earlier
stream.started_s field or from the eventual result.

At that observation point, the legacy success predicate is frozen before any
cleanup cancellation: FINISHED observed and no non-CANCELLED transport error.
The old StreamHandle does not put typed in-band errors into snap.error, so those
raw codes remain recorded without silently changing this literal predicate.
FINISHED without EOF can pass the frozen legacy check. A FINISHED frame arriving
only after cancellation cannot repair a frozen failure. The owned cleanup
separately cancels outstanding work, waits for the actual consumer-exit event,
and preserves final records alongside the frozen snapshot. Cleanup failure
remains a cleanup failure, not a rewritten business result.

Fixtures cover open taking 2s followed by completion 29s after open, FINISHED
without EOF at observation, FINISHED only after cancellation, and typed-error
plus FINISHED. Flow scheduling is a deterministic simulation, while all profile
probe/flow/recovery RPC consumers use actual RecordedRequests and response-based
Fetch/Generate selection, including identical NON_BATCH shape and role copying.
This establishes Python construction behavior, not real Java acceptance.

All eight profile/grade configurations are compared in full with the old factory:
fault preset, topology, both cache pools, dynamic discovery, PRIORITY and omitted
queue timeout. Existing exact topology checks and first-failure blocking remain
declared strengthening. New handlers aggregate through elastic_lifecycle; no
shared elastic.py, catalog or Java change is required. Independent static and
fixture review passed for `870ff40e27466b8e252881ebfc82dccd8806781a`: the reviewer
independently ran 114 elastic tests in 31.462s and eight added-worker tests in
0.455s against core4e, verified default-catalog compilation of all eight
57-stage/75-check variants, and confirmed nine unrelated plans remain unchanged
with exactly 12 new unique legacy-ID/profile pairs. Real Java acceptance remains
pending. Legacy cases and historical results remain.
