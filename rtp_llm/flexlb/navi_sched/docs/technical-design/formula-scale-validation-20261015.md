# Formula validation handoff, 2026-10-15

## Scope and evidence levels

This phase changed only two test files in the detached integration tree
`/tmp/navi-726-gate-isolation-20261015`, based on
`7265da5deef3f52ac5b4cf5e708ec423422bc0af`, plus this report. Production
formulae, defaults, the original worktree, branch pointers, online traffic and
configuration were not changed. No commit, push, rebase or deployment was run.
Existing dirty changes were not replaced.

Evidence directory (E below):
`/tmp/alibaba-bazel-test/navi-integrate-20261015`.
Current source fingerprints and historical source references are in
`E/report-lineage.json`; current fingerprints are NOT historical execution hashes.
The inventory covers 11 experiment reports and 3 incident reports. Script names
extracted from report text are search candidates: a missing file means not found
inside this NaviSched tree, not proof it never existed in another repository.
Brace/wildcard paths in that JSON are literal references, not expanded existence
checks. The six-formula 15:38-15:53 repair evidence remains unchanged.

## Full 19-parameter comparison: verified locally

New cases:

- `resource/NonLinearPrefillModelRTest.cpp`,
  `testFull19ParameterFixedSampleAndJacobian`.
- `test/CompareLatencyOnlineRlsTest.py`,
  `test_full_19_parameter_fixed_sample_and_jacobian`.

All 19 weights are nonzero, including batch-count P6 and decode P17/P18.
All 16 production normalized input components are positive. The raw fixture is
`[2,3,1024,2048,4194304,2097152,4,512,3072,9437184,1572864,5,250000,6,8192]`.
P17=7, P18=11; generated decode tokens are normalized by 1024, not 512.
Both C++ public inference and C++ AutoGrad, and the existing full Python RLS
formula, assert `67100.84583536643 ms` with absolute tolerance `1e-8 ms`.
Both languages compare all 19 analytic derivatives with the same independently
calculated decimal reference and with central finite differences. Analytic
reference tolerance is relative `1e-10` (C++ uses max(1,abs(reference)) and
Python also permits absolute `1e-10`); difference tolerance is
`1e-6 * max(1, abs(derivative))`. Each decode weight is separately disabled as
a counterexample and must strictly reduce this fixture's prediction.

The independent reference uses 70-digit Decimal arithmetic, no model imports,
and central differences of size `1e-15`: `E/fixed19-reference.py` and
`E/fixed19-reference.json`. Its unrounded prediction is
`67100.84583536642833825067137881095574085665586640997012445559536891420`.

Execution chain:

1. `E/full19-final.log`: C++ 24 and Python 25 actual cases pass.
2. Deliberately change only the new fixture's decode count 6 -> 5, keeping the
   independent reference fixed. C++ filtered case fails, Bazel exit 3:
   `E/full19-negative-cpp.log`, profile `E/full19-negative.profile.json.gz`.
   Python filtered case fails, exit 1: `E/full19-negative-python.log`.
3. Restore decode count to 6. `E/full19-restored.log`: C++ 24/24, Python 25/25,
   two targets executed, exit 0; profile `E/full19-restored.profile.json.gz`.

This is test-fixture fault injection, NOT removal/reapplication of a production
repair. There was no new production repair in this phase. It does not replace
the six earlier tools' old-fail/fix/revert-fail/restored evidence.

Final affected test command, from the isolated tree:

```bash
bazel --output_base=/tmp/alibaba-bazel-test/navi-726-gate.8Hr4K0/output_base test \
  //aios/navi_sched:nonlinear_prefill_model_r_test \
  //aios/navi_sched:compare_latency_online_rls_test \
  --config=clang --config=new_abi --config=debug --config=py310 \
  --cache_test_results=no --test_timeout=900 --test_output=all \
  --profile=/tmp/alibaba-bazel-test/navi-integrate-20261015/full19-restored.profile.json.gz
```

Only these test targets were affected by this phase. The previously qualified
six-tool, operations, role and replay tests were not needlessly rerun. This
fixture is a non-PD formula check, not a Master/Agent transport or live PD check.

## Replay executable and libpython: verified locally

`E/replay-runtime-audit.py` independently invokes the existing linked executable
on one symlinked retained real MI308 JSONL input. It removes LD_LIBRARY_PATH and
LD_PRELOAD from the child environment; no new symlink/library workaround or
system installation was added. Complete argv, resolved binary, input SHA256,
exit codes and output tag counts are in `E/runtime-audit.json`.

Command:

```bash
/tmp/navi-726-gate-isolation-20261015/bazel-bin/aios/navi_sched/latency_online_replay \
  /tmp/alibaba-bazel-test/navi-integrate-20261015/runtime-one-input \
  /tmp/alibaba-bazel-test/navi-integrate-20261015/runtime-real-output.tsv \
  4 1 1 0 all 0 fixed
```

`E/runtime-real.log` records exit 0. Output is NOT empty: P=2345, U=587,
Z=587, R=9382. This is an execution smoke with explicitly fixed rates and
residual off, not an accuracy experiment or exact online trajectory claim.
No-argument invocation returns usage and exit 2 (`E/runtime-invalid.log`),
consistent with the source's argument-count branch.

`E/runtime-ldd.log` has no missing libraries and resolves libpython3.10.so.1.0
to the binary's `navi_sched/libpython3.10.so.1.0` sibling.
`E/runtime-dynamic.log` records NEEDED libpython3.10.so.1.0 and
RUNPATH `$ORIGIN/navi_sched`; `BUILD`'s `wheel_python_runtime` genrule produces
that sibling library. This establishes correct resolution in this build tree,
not a system-wide libpython installation or proof of a container image's layout.

Audit-driver failures are distinguished from executable results: the first
invocation used system Python 3.6 and could not accept `capture_output` (its
driver log was overwritten on retry; the error was observed in the tool output).
A Python 3.10 invocation
then executed all commands but the audit incorrectly expected usage exit 1.
The driver expectation was corrected to the actual code contract, exit 2.
The successful executable run was NOT repeated: `E/runtime-saved-result-verification.log`
independently validates its saved raw results. `E/runtime-audit-driver.log`
therefore remains an audit assertion failure, not a success log. This has no
bearing on the separately recorded real executable exit 0 and nonempty output.

## Old-data coordinate countercheck: verified, not a refit

`E/legacy-window-audit.py` reads all retained A/B JSONL files, follows every
logged parameter update, and compares old physical-coefficient predictions to
`base_prediction_before_ms`, without residual, fitting or hyperparameter search.
Each input's hash, sample count, event range and state continuity result are
in `E/legacy-window-audit.jsonl`; summary is `E/legacy-window-summary.jsonl`.

| Window / hardware | Nodes | Samples | Maximum absolute formula/log difference, ms |
| --- | ---: | ---: | ---: |
| A / MI | 10 | 25447 | 5.238689482212067e-10 |
| A / RTX | 9 | 17429 | 5.238689482212067e-10 |
| B / MI | 10 | 67805 | 5.311449058353901e-10 |
| B / RTX | 10 | 46468 | 5.165929906070232e-10 |

All 157149 samples have 17-parameter states; skipped non-17 states=0 and
update-before/previous-after mismatches=0 (absolute state tolerance 1e-9).
This proves base-forward/log agreement, NOT equality of refitted optima,
residual state, loss weighting or experimental efficacy.

Actual aggregate event windows (CST):

- A `/tmp/navi-rate-config-aware-0106`: 2026-10-06 03:41:12.901064 to
  04:38:19.317946.
- B `/tmp/navi-independent-combined-original-0106`: 2026-10-04 22:41:20.594823
  to 2026-10-05 01:55:54.048063.

These A/B labels are NOT the A/B/C windows in the Oct 7/8 online-controller
reports. Four focused low/high-pressure checks are also in
`E/legacy-{A,B}-{mi,rtx}.jsonl`. They show large differences if those old
physical weights are wrongly interpreted as current scaled coordinates.
For reuse the conversion is coordinate weight = physical coefficient / scale;
optimizer state, physical bounds and fitting setup must also be audited.

Historical committed source `f05b8b16eca` is preserved as
`E/phase-script-f05b8b16eca.py`: its base formula has no per-weight scale.
This is a direct counterexample to declaring every old numerical report wrong.
The Oct 11/14 real-mi/rtx evidence from the preceding phase instead uses current
scaled parameters with zero decode weights: `E/real-mi-audit.jsonl` and
`E/real-rtx-audit.jsonl`. Do not merge these data windows or call them a full
19-parameter real-data validation.

## Per-report disposition

The table distinguishes historical observations from reuse as a current-model
optimum. A report's last Git commit is provenance for repository content, NOT
proof of the exact dirty script executed to produce each table. The JSON
inventory records current script hashes, dependencies and last tracked commits;
untracked reports/scripts have no such revision and remain explicitly unknown.
No old benchmark tables were recomputed or mechanically edited in this phase.

| Report | Script/version and sample evidence | Disposition |
| --- | --- | --- |
| `test/LatencyPhaseLeastSquaresReport.md` | Last tracked report d36595a1f3f; `AnalyzeLatencyPhaseLeastSquares.py`, old 17-parameter physical v21; A/B windows above. Its documented counts match this audit exactly. `AnalyzeLatencyFormulaCapacity.py` reference is not present under the isolated NaviSched tree. | Preserve historical formula/log agreement and stated old-window observations. New scaled/constraint/residual-off optimum requires a new fit. Missing capacity-script provenance prevents declaring every section reproducible. |
| `test/LatencyPressureCurvatureAblation.md` | f05b8b16eca; `AnalyzeLatencyPressureFormula.py` and shared phase module; old physical v21, same A/B. Twelve `/tmp/navi-pressure-formula-{current,independent}-*.jsonl` outputs remain. | Preserve old-window results as historical, supported by committed old formula and raw-data agreement. Invalidate any rerun using pre-fix pressure script with the new scaled base; re-fit for current comparisons. |
| `test/LatencyGlobalLeastSquaresVsOnlineReport.md` | Untracked report, producing execution hash unknown; `CompareLatencyGlobalLeastSquares.py` -> phase module; A/B; `/tmp/navi-global-ls-{current,independent}-{log,raw}.json` retained. | Physical old-window baseline supported; historical fitted scores are not independently regenerated here. Do not invalidate merely because coordinates changed, and do not reuse their optimum as today's configuration. |
| `test/LatencyFixedCurvatureComparison.md` | Untracked report; `CompareLatencyFixedCurvature.py` -> phase module; A/B; `/tmp/navi-fixed-k-{A,B}-final.json` retained. | Historical results only, execution hash unknown. Current tool needs a separate scale-contract check: convex path delegates to scaled base, concave path uses unpacked weights directly, and fixed curvature is inserted without physical-to-coordinate conversion. Passing self-consistent Jacobian tests does not establish a common physical comparison. Not fixed in this phase. |
| `test/LatencyNonlinearShapeDiagnosis.md` | Untracked report; SingleIndex, ShapeEstimators, Square, Augmented, Nested, QuarticRoot, QuarticPolynomial and shared phase modules; A/B. `/tmp/navi-single-index-{A,B}-d{1,2,4}.json`, `/tmp/navi-shape-estimators-{A,B}.json` retained. | Preserve explicit statistical/causal caveats. Current scaled rankings based on the six pre-fix tools must be discarded/recomputed. Historical physical rankings are not proven false by this repair, but producing hashes are absent. ShapeEstimators/SingleIndex also need current curvature-coordinate auditing. |
| `test/LatencyTransitionWidthAndRisingReplay.md` | Untracked report; TransitionWidth + RepeatRisingReplaySamples + EvaluateRisingRepeatReplay + C++ replay. A/B; `/tmp/navi-transition-{A,B}-{blocks,forward}.json` and A 2400-evaluation outputs retained. | Separate static width results from C++ repeated-sample interventions. Discard pre-fix width results applied to scaled coordinates. Preserve historical limited-prefix/paired-sample caveats; current replay efficacy requires current inputs and baseline equality, not this smoke. |
| `test/LatencyFrozenOnlineAblation.md` | Untracked report; `CompareLatencyFrozenState.py` invokes replay binary. Its A is Oct 7, B/C Oct 8, NOT this audit's A/B; `/tmp/navi-freeze-{A,B,C}-20261008.jsonl` and anchor summaries retained. Producing binary/hash unknown. | Not directly invalidated by the six Python formula repairs. Historical freeze comparisons only; cannot use to claim residual is necessary after new scale/default changes. Need original binary provenance and matched baseline for current reuse. |
| `test/LatencyPressureBucketAblation.md` | Untracked report; `CompareLatencyPressureBuckets.py` invokes replay; Oct 7/8 A/B/C, retained `/tmp/navi-pressure-bucket-{A,B,C}-20261008.jsonl`; logged rates differ by window. | Not directly invalidated by six formula fixes. Retain admission-pressure/fit-inflight distinction; absent original execution hashes means efficacy not newly certified. Current history/rate experiments must be requalified. |
| `test/LatencyErrorRateLearningRateAblation.md` | ce518547db4; C++ `LatencyOnlineReplay.cpp`; Oct 7/8 A/B/C and later controller subsections, not the phase A/B audit. Some old TSVs were intentionally deleted in disk cleanup; report references alone do not prove retained complete traces. | Preserve as historical offline evidence, not a current tuned optimum. Original binary, effective candidate-rate overrides and retained traces must be individually verified before reuse. The six fixes do not establish all older learning-rate searches were valid or invalid. |
| `test/LatencyChirp20261008Investigation.md` | d351565bb474; C++ replay and joined Master/Agent/vLLM evidence; Oct 8 plus later decode experiments. The report has multiple windows/configurations, not one A/B corpus. | Retain raw paired timing/step observations subject to source availability; no new causal certification here. Decode parameter-rate rankings cannot be carried to scaled/nonnegative/residual-off defaults without renewed checks. |
| `test/LatencyFeatureNeighbors20261009.md` | 34da06ec495; FeatureNeighbors, Drain and RequestLifecycles; Oct 9 17:15-17:45 focus and later separate drain/lifecycle windows. Normalization of feature distances is not weight scaling. | Not directly invalidated by six formula fixes: raw feature/latency conflicts and lifecycle observations are separate evidence. Recomputed model predictions need the producing formula and revision. No claim of fresh all-pair/lifecycle verification. |
| `docs/technical-design/incident-20261010-2100.md` | Last tracked 55462e2ce8a, with later dirty additions; multiple external temporary audit scripts and Oct 10+ windows, see inventory. | Six repairs neither prove nor disprove the 21:00 causal chain. Preserve explicit missing Master same-batch/candidate/release evidence. Re-evaluate individual numerical claims only when their exact script/data/rate provenance is recovered. |
| `docs/technical-design/mi308-peak-trough-20261012.md` | 7d90b53a544; external analyze_wave_stages/steps; Oct 12 origin 20:12:40 CST; raw Master/Agent/vLLM/KMon timing windows. | Raw timing/routing analysis is outside six offline-formula changes; preserve as window-specific evidence, not current-model efficacy or a scale root-cause proof. |
| `docs/technical-design/mi308-peak-routing-20261013.md` | bc281e7026a; external peak/lifecycle/release scripts; Oct 13 04:27-05:02 CST rounds. | Preserve explicit coverage and timing caveats. No new verification of its proposed mechanisms or complete same-batch reconstruction was performed in this phase. |

Additional read-only scope warning: `PlotLatencyShapeOperatingRange.py:21`
still directly multiplies serialized weights without scale. This is appropriate
for a proven physical-coefficient old output, not automatically for a new scaled
output. `CompareLatencyShapeEstimators.py` and `CompareLatencySingleIndex.py`
mix shared helpers and direct curvature access; these are unresolved current
tool contracts, not diagnosed online incident roots. Do not call the entire
analysis suite scale-safe because the six repaired suites pass.

## Remaining work and independent counterchecks

- Verified: full nonzero 19-parameter local forward/gradient fixture; deliberate
  wrong-fixture detection and restored affected targets; independent executable
  run with actual nonempty output; ELF/libpython resolution without loader-env
  workaround; old A/B raw coordinate/log agreement over all 39 nodes.
- Missing: exact generating script/binary hashes and full trace availability
  for several historical reports, particularly untracked and mixed-window ones.
  The per-report mapping is complete as an inventory, not a claim that this
  provenance gap has been recovered.
- Missing: dedicated physical-unit contracts for fixed-curvature/shape/plot
  downstream tools. No new formula or default changes were made to cover them.
- Missing: broad/random full-19 real-stream online/offline equality, full
  Master/Agent admission snapshot alignment, deployment-image runtime check,
  and actual load-tested accuracy. No current optimization benefit is claimed.
- Earlier six-tool old-fail/fix/revert-fail/restored tests remain the repair
  chain. This phase adds coverage, not a new production root cause or repair;
  no live validation was performed.

Independent checks deliberately distinguish prediction/log equality from
prediction/target error; old physical coefficients from current coordinates;
batch counts from QPS phase; explicit zero-decode old real samples from nonzero
decode fixed fixtures; and successful executable exit from a failed audit-driver
assertion. None of these offline results resolves the online 21:00 or MI308
peak error causal investigation on its own.
