# Downstream scale-contract audit, 2026-10-15

## Scope and evidence level

This is a local offline-tool correctness repair, NOT an online incident root
cause, a new model experiment, or evidence of improved load-tested accuracy.
Worktree: `/tmp/navi-726-gate-isolation-20261015`, detached base
`7265da5deef3f52ac5b4cf5e708ec423422bc0af`.
Evidence directory E: `/tmp/alibaba-bazel-test/navi-integrate-20261015`.
No production C++ formula/default, main worktree, branch pointer, online
configuration or traffic was changed. No commit/push/rebase/deploy was run.

Followup to `formula-scale-validation-20261015.md`: that report's downstream
missing-work list describes the earlier phase; the nine tools below now have
local contract coverage. Historical producing hashes remain unknown where
previously unknown. The current tree DOES contain
`test/AnalyzeLatencyFormulaCapacity.py` (mtime 2026-10-15 13:42); the earlier
report's claim that it is absent is incorrect for the current tree. Existence
does not establish the historical execution hash or validate its old results.

## Contract and per-file repair

Serialized/fitted weights remain coordinates w. Physical coefficients are
w * PARAMETER_SCALES, so d(prediction)/dw includes the same scale. Helpers
`inner_input` accept coordinate weights; `operating_masks` accepts physical K.
No production constraints/defaults were introduced or changed.

| File under test/ | Direct defect and correction | Distinguishing test |
| --- | --- | --- |
| CompareLatencyFixedCurvature.py | Physical fixed K was inserted as coordinate K; divide by scale8. Concave branch lacked physical coefficients and Jacobian chain factors. Report K physically. | Inverse identity, C++ golden forward, both concave physical references and all-active finite differences. |
| CompareLatencySquareShape.py | Raw coordinates used as physical coefficients/Jacobian; existing physical P15>=1 bound represented as coordinate 1. | Physical forward, finite differences, exact bound*scale=1. This preserves an existing offline constraint, not a new production bound. |
| CompareLatencyShapeEstimators.py | Scaled u combined with unscaled bypass P7/P9/P15 and K. | Exact adjusted target and physical K passed to region helper; fitting mocked. |
| CompareLatencySingleIndex.py | Region K unscaled. | Exact physical K at actual evaluation caller; no neural training experiment. |
| CompareLatencyCrossWindow.py | Region K unscaled. | Exact physical K at score caller. |
| CompareLatencyFeatureCapacity.py | Region K unscaled. | Exact physical K at evaluation caller; fitting mocked. |
| AnalyzeLatencyFeatureGroups.py | Region K unscaled. | Exact physical K at evaluation caller; fitting mocked. |
| PlotLatencyHighPressureU.py | Scaled u mixed with unscaled bypass, zero anchor, K, boundary and title. | Forward reconstruction, adjusted target, actual axvspan coordinates. |
| PlotLatencyShapeOperatingRange.py | Duplicate inner helper omitted scales; curve used coordinate zero anchor/K. | Reuse existing inner helper; actual original/square matplotlib line values match physical reference. |

Two existing tests, FixedCurvatureTest and SquareShapeTest, now construct their
physical fixtures as coordinate=physical/scale. Existing physical expected
values/tolerances are retained, not mechanically relaxed. New
`LatencyDownstreamScaleContractTest.py` contains 12 cases. BUILD adds the exact
`//aios/navi_sched:latency_downstream_scale_contract_test` target with explicit
sources; existing unrelated BUILD changes are preserved.

## Dependency inventory and bounded exclusions

`E/downstream-dependencies.json` records import symbols, consumer paths and
line numbers for all static Python imports under aios/navi_sched: 62 direct
module edges from the changed modules plus shared phase/inner helpers, and 23
transitive edges. These are conservative module dependencies, not a claim that
every imported symbol changed. Dynamic/external callers are not proven absent.

Direct consumers of repaired functions are the nine entries above and their
tests. `CompareLatencyTransitionWidth.py` imports only FixedCurvature's
unchanged `segment_masks`; its three tests are included in the restored run.
Shared `inner_input` also feeds AugmentedSquare, QuarticRoot, QuarticPolynomial
and their tests. Root/polynomial already convert temporary physical arrays back
to coordinate weights before calling it; no additional edits were made. Their
affected tests are included, as are GlobalLeastSquares aggregation/segments
and the shared phase formula tests. The five operating-region consumers are
tested separately, so one failure cannot hide another caller.

The wider shared-base graph includes CycleGeneralization, FormulaCapacity,
BootstrapHeldComparison, LinearLeastSquares, OperatingMargin,
EvaluateDecodeProgressFeature -> AnalyzeVllmDecodeSchedulerState ->
EvaluateVllmDecodeCohortFeatures. They do not call these nine repaired functions;
this inventory is NOT full scale certification of every sibling experiment.
In particular OperatingMargin contains direct unpacked ratio/Jacobian math
(`operating_ratio`, `margin_residual_and_jacobian`) and DecodeProgress contains
its own forward math; their historical/current scale contracts need a separate
bounded audit before reuse. No new experiments or unrelated repairs were made.

## Executed repair validation chain

1. New tests on the saved old implementations fail:
   `E/downstream-old-fail.log`, then separately covered caller cases in
   `E/downstream-old-fail-all-entries.log` (10 actual cases).
2. Fixes pass 17 standalone cases (10 new + Fixed5 + Square2):
   `E/downstream-fixed.log`. Two actual plotting cases are then added;
   Bazel runs 12/12, exit0: `E/downstream-fixed-bazel.log` and
   `E/downstream-fixed.profile.json.gz`.
3. Temporarily remove ONLY this phase's nine implementation fixes using
   pre-fix snapshots, keeping the corrected/new tests. Bazel executes 12
   cases, fails 13 assertions including concave subcases, exit3:
   `E/downstream-reverted-bazel.log`, `E/downstream-reverted.profile.json.gz`.
   This is actual test failure, not build failure, timeout, zero tests or cache.
4. Restore exactly these fixes. Bazel executes 12/12, exit0:
   `E/downstream-restored-bazel.log`, `E/downstream-restored.profile.json.gz`.
5. Final affected standalone run executes 46/46, exit0:
   `E/downstream-affected-final.log`. Suites: DownstreamScaleContract12,
   FixedCurvature5, Square2, HighPressurePlot1, SingleIndex2, Augmented3,
   QuarticRoot4, QuarticPolynomial3, TransitionWidth3, GlobalLeastSquares4,
   PhaseLeastSquares7. Scientific interpreter: `/opt/conda310/bin/python3.10`.

The Bazel runs use isolated output_base
`/tmp/alibaba-bazel-test/navi-726-gate.8Hr4K0/output_base`, explicit
`--config=clang --config=new_abi --config=debug --config=py310`,
`--cache_test_results=no --test_timeout=900 --test_output=all` and distinct
profiles. Relevant previous C++/full19 tests were not rerun: their sources are
unchanged in this phase, and prior qualified logs remain cited by the earlier
report. Tests verify 17-parameter downstream math; they do not impersonate a
full19 real-stream check.

## Independent physical reference and real retained samples

Existing C++ `NonLinearPrefillModelRTest.cpp` asserts the same 17-parameter
fixture value 66741.79706653491 ms as the new fixed-curvature test and the
shared Python test. Previous full19 C++/Python golden
67100.84583536643 ms and Jacobian evidence remain unchanged. Concave/square
references are independently calculated physical equations, compared to
analytic derivatives by scipy finite differences across all active coordinates.
Inverse fixed-K mapping is checked separately from forward equality. Actual
matplotlib line/region data are checked, not just successful save calls.

`E/downstream-real-audit.py` replays logged parameter states only, without
fitting or changing rates/residual. Full source SHA256, event microsecond
ranges, counts and numerical differences are in
`E/downstream-real-audit.jsonl`; exit0 is in `E/downstream-real-audit.exit`.
Input source names are recorded verbatim, not inferred from report labels.

| Retained stream | Total samples / updates | Low count R+W<=20 | High count R+W>=100 |
| --- | ---: | ---: | ---: |
| MI308 11.225.195.118, residual-off controller input | 2360 / 587 | 718 | 1212 |
| RTX5000 215.91.17.72, exact-cycle input | 1296 / 321 | 220 | 801 |

Both streams: update-before/previous-after mismatches=0 (atol1e-9), skipped
nonzero-decode states=0. These are 19-element states with decode weights zero;
downstream checks deliberately select17, not a full19 real workload claim.
For reported low/high groups: max independent base/log difference
5.238689482212067e-10 ms; max base/Python or fixed-convex difference
2.9103830456733704e-11 ms; inverse max8.881784197001252e-16 coordinate;
square physical max1.862645149230957e-9 ms; concave max1.4551915228366852e-11 ms;
plot/shared u difference0; region classification mismatch0.
All samples, including intermediate pressure, passed explicit absolute1e-7
checks. These variant differences are physical-formula comparisons, NOT errors
against observed target latency or evidence of variant improvement.

Audit-driver counterexample: its first run incorrectly wrote base shape as
u*(u+sqrt(u*u+K)), and failed immediately by629.88 ms. Source code independently
shows shape=u+sqrt(u*u+K); the audit reference was corrected, NOT production
code or tolerances. Original failed stderr is retained in
`E/downstream-real-audit.stderr`, corrected run in
`E/downstream-real-audit-restored.stderr`. This prevents an audit implementation
error being mislabeled a production regression.

## Historical report disposition and independent counterchecks

Per-report original script/data/hash availability remains recorded in
`E/report-lineage.json` and the earlier report's 14-row table. This followup
does not invent producing hashes or recompute old benchmark tables.

- FixedCurvatureComparison: current scaled runs using the pre-fix fixed-K or
  concave paths must be discarded/recomputed. Historical physical A/B results
  remain historical with unknown producing execution hash, not newly disproven.
- NonlinearShapeDiagnosis: current scaled square/shape-estimator bypass and
  region rankings need recomputation, in addition to six earlier formula
  repairs. SingleIndex training is unchanged; physical region labels, not its
  entire prediction model, were repaired. Plotted conclusions require known
  input coordinate convention and refreshed curves.
- PhaseLeastSquaresReport, PressureCurvatureAblation and GlobalLeastSquares:
  retain earlier old-physical-data caveats. This phase changes neither the
  shared phase formula nor those original fitted optima.
- TransitionWidthAndRisingReplay: segment_masks import unchanged; earlier
  width repair/recompute caveats stand. No repeated-replay efficacy claim.
- FrozenOnlineAblation, PressureBucketAblation, ErrorRateLearningRateAblation:
  C++ replay results are not automatically invalidated by these plot fixes;
  original binary/rate/source gaps still prevent claiming current optima.
- ChirpInvestigation, FeatureNeighbors and the three incident/MI308 timing
  reports: raw timing/feature evidence is separate. No new incident causal
  certification; transformed model/region numbers require provenance checks.

Independent audit distinguishes coordinate from physical coefficients, a
fixed C++ golden from a new C++ run, module dependency from actual changed
function calls, zero-decode streams from full19 fixtures, formula/log equality
from model/target accuracy, and two different windows from a controlled A/B.
The real streams are not simultaneous cross-hardware pressure experiments.
Their full event ranges and hashes are retained, so group selection cannot be
silently called all nodes/all pressure or a matched-window efficacy test.

## Handoff and remaining release gates

`E/downstream-file-status.json`: per-file before/after SHA256 and status.
`E/downstream-phase.patch`: complete this-phase changes relative pre-fix
snapshots, including new tests, BUILD target and this report. Untracked tool
files are included; ordinary git diff alone would omit them. Earlier integration
changes are NOT implicitly included in this phase patch.

Locally ready for review: nine tool repairs, two corrected fixture files, new
12-case test, BUILD target and this report. Original integration's full pending
feature set still requires its own complete staging/review and final combined
tests; do not ship this scoped patch while claiming all historical dirty
features are integrated. Original reports/tools are preserved.

Not done: commit/push/CI/package/container verification, fresh Master/Agent
same-batch alignment, full19 real-stream equality, online deployment and stable
load-tested accuracy comparison. No real-environment verification was run.
No residual/learning-rate tuning or new model search was performed. A statement
that these fixes improve MI308 peaks would need new controlled online evidence.
