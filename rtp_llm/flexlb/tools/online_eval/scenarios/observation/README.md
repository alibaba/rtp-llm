# Read-only observation adapters

`scenario.actions.observation.HANDLERS` supplies `snapshot` and `observe`; the
core registry explicitly installs these descriptors. Both return registered
handles, never raw paths or user-created resource identifiers.

`snapshot` accepts a nonempty `sources` list: `master_debug`, `engine_snapshot`,
`engine_requests`, and `client_records`. Client records require a typed `flow`
or `requests` reference whose resource implements ClientRecords v1
`snapshot_records()`. Master debug additionally requires an environment with
`debug_enabled: true` and the debug-enabled Java artifact. It stays disabled by
default. `targets` names expected engines; absent targets are partial evidence,
and raw responses remain intact. Membership changes are explicit; mock HTTP
provides no restart generation, so `engine_generation` stays null.

Each sample records its source, environment epoch, monotonic capture interval,
wall-clock timestamps, raw response and errors. Master instance changes, busy,
truncation, unavailable/not-applicable components and source failures are not
complete evidence. `required` defaults to true and produces CheckResult.ERROR
for incomplete coverage. With `required: false`, the fixed `sources` check only
confirms optional collection; its PASS does **not** assert source availability.
The artifact retains partial/error status for any dependent checker.

`observe` supports `window` with `duration_s`, `start` with `max_duration_s`, and
`stop` with an observation reference. Sampling interval defaults to 0.5 seconds,
with bounds of 10,000 samples and 32 MiB, including a final dataset size check.
The registered cleanup cancels and joins only its own thread. The worker writes
`worker_exit_mono` and sets its independent completion event in `finally`, after
its last sample or error write. Stop requires that event and exit record before
freezing, even if `is_alive()` reports false. A missing completion signal at the
cleanup deadline is TIMEOUT, not successful cleanup. Repeated stop returns
the same handle, frozen dataset and artifact. After an environment epoch change,
stop joins the old observer but never samples the replacement environment.

Cohort bases are `issued_in_window` (default), `submitted_in_window` (Schedule
start) and `terminal_in_window` (transport terminal). Their interval is
[start, stop). Final freezing reads the same pinned record provider so attempts
issued after the last periodic sample remain present, including unfinished
attempts. It does not wait for cohort settlement (`cohort_settle: not_waited`).
Transport completion is not business completion. Source coverage alone does not
assert a nonempty or successful business cohort; scenario-specific checks must
do that. Frozen resource `to_dict()` returns an independent deep copy.

`terminal_cohort.yaml` exercises deferred streams across start/wait/stop and
checks the ordinary request completion/error outputs. It demonstrates source
collection, not a no-Fetch lifecycle proof. The separate Python case
`normal_no_fetch_observation` uses a fresh isolated environment, successful
Schedule-only request, nonempty Prefill accepted/completed evidence, all-engine
FetchResponse counter delta zero, then recovery outside that window. It does
not inject status suppression, shorten TTL, or infer C++/GPU/connector ownership.

## Validation

Against core commit `3e192c94d4`, both YAML profile instances compile. The local
suite passes 24 core and 12 observation tests, including execution through the
real compiler/runtime with a fake backend, frozen nonempty terminal cohorts,
required-source ERROR under a finding, and optional-source check contracts.
The eight debug-client tests also pass. This does not claim remote YAML-backend
execution or full-suite registration; those are the framework owner's integration
steps.

The normal no-Fetch case passed on isolated lease `agent5_nofetch_20260907`, host
111, using 2 Prefill and 4 Decode mock engines. Request `7301200001` was accepted
via EnqueueBatch on `prefill-0`, then completed. Twelve samples preserved zero
FetchResponse deltas for every engine; the final Master components were complete.
Recovery request `7301200002` passed outside the no-Fetch window. Both owned Java
processes exited, all ports 61020–61039 were bindable, and the lease was released.
The case does not assert that every owner clears: it preserves the separate
scheduler, queue, Prefill, Decode and engine rows for interpretation.
