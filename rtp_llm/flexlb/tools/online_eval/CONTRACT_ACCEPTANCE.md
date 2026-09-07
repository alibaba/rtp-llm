# Independent scenario acceptance

`flexlb_ft.acceptance` audits normalized JSON without starting engines. Run it
from this directory with `python3 -m flexlb_ft.acceptance`. This is an independent
oracle, not a scenario compiler or executor. `flexlb_ft.scenario_acceptance`
adapts the core v1 plans and results independently of the runtime module.

The adapter provides `normalize_plans(plans, check_contracts=None)`,
`normalize_results(plans, results)` and `expand_coverage(document, inventory)`.
Check contracts are keyed by instance ID then qualified check ID, and can supply
reviewed `numeric`/`min_samples` fields. They cannot override finding/check identity.
For non-core checks, evidence requires explicit `complete: true`; sampled metrics
also need `sample_count`. An empty evidence dictionary is not complete observation.
Core typed boolean/integer comparisons are scalar contracts rather than sampled
metrics. Callback success is labeled `reported_callbacks_only`; it does not attest
that remote processes or sockets are gone. Unknown/missing explicit coverage
variant/profile targets reject rather than silently disappear during expansion.

## Freeze the actual source

```bash
python3 -m flexlb_ft.acceptance snapshot --revision "$(git rev-parse HEAD)" > /tmp/legacy-contracts.json
```

The snapshot imports current `ALL_CASES` and records execution order, category,
declared and effective profiles, requires, expected-fail, source, and a metadata
+ function AST digest. It does not start a JVM. Supply the actual revision and
record dirty status separately; the revision argument is provenance supplied by
the caller, not an attestation. Re-freeze after accepted additions. Never derive
coverage from the historical 131-case count or a proposed scenario-file count.

AST digests anchor the source being claimed. They do **not** prove semantic
equivalence of a new implementation, capture helper/global changes, or replace
comparisons of actual thresholds, sample windows, effective configuration and
per-check results. Keep those paired artifacts with the review.

## Normalized coverage input

Inventory has `instances`, with each row containing:

```json
{
  "id": "scenario::variant::profile",
  "scenario_id": "scenario",
  "variant_id": "variant",
  "profile": "batch-window",
  "requires": ["enqueue_batch"],
  "stages": ["setup", "observe"],
  "checks": [{"id": "observe.latency", "numeric": true, "min_samples": 3}]
}
```

`checks` must come from declared check IDs, never inferred from action names or a
successful parse. A known finding has a `finding_id` on its specific check. A
retained Python invocation also supplies `backend: "legacy"` and
`legacy_case_id` matching the original case.

Coverage has `legacy_cases` as an ordered list, one row per frozen name:

```json
{
  "id": "old_case",
  "disposition": "migrate",
  "targets": [{
    "instance_id": "scenario::variant::profile",
    "check_ids": ["observe.latency"],
    "contract_digest": "digest from frozen source"
  }]
}
```

`retain_legacy` additionally requires a nonempty `rationale`. It is counted
separately and does not count as migrated coverage. Every effective legacy
profile must have a target. New independent instances are allowed; unknown
legacy mapping names and duplicate IDs are rejected.

```bash
python3 -m flexlb_ft.acceptance coverage /tmp/legacy-contracts.json /tmp/inventory.json /tmp/coverage.json
```

The scenario compiler's proposed dictionary-form `coverage.yaml` can be adapted
by expanding scenario/variant/profile targets into explicit instance IDs, checking
actual handler check declarations, and attaching the frozen source digest. Do not
treat this expansion or a matching digest as execution evidence.

## Result and resource evidence

Result input has `instances`, each with `id`, `errors`, `cleanup_errors`,
`leaked_resources`, `stages: [{id,status}]`, and
`checks: [{id,status,evidence_complete,value,sample_count,failure_kind}]`.
The three error/resource fields must be present lists. Numeric checks require a
finite number and sufficient integer samples; a zero with samples is valid,
missing data is not. `FAIL` is a confirmed finding only on a declared finding
check with complete evidence and `failure_kind: "contract"`. Setup failures,
TIMEOUT, ERROR, cleanup errors and BLOCKED do not become findings. A stage FAIL
must be explained by an actual failed check. All required stages must execute.

```bash
python3 -m flexlb_ft.acceptance results /tmp/inventory.json /tmp/results.json
python3 -m flexlb_ft.acceptance lifecycle /tmp/resource-events.json
```

Lifecycle input is an event list with `action: acquire|use|cancel|release`,
`resource_id`, and `epoch`; an optional `stage` documents where it happened.
Resources can survive stage boundaries. Cancellation is not release, and a stale
epoch release cannot clear a newer resource. Partial setup must release resources
already acquired. This validates reported ownership events, not the actual state
of processes, sockets or remote engines. `audit_selection(serial, parallel)` also
compares expanded instance contracts while allowing lane assignment to differ.

## Integration acceptance still required

- Adapt the stable compiler's instance/check declarations, coverage mapping and
  `ScenarioResult` to these inputs; preserve errors, typed evidence and BLOCKED.
- Execute fake backend fixtures against the real runtime: wrong/forward typed
  refs rejected before setup; stale epochs rejected; flow survives stages;
  partial setup and stage TIMEOUT run independent-budget LIFO cleanup; canceled
  handles are joined; cleanup errors/leaks remain in results. Oracle mutation
  tests alone do not establish these runtime properties.
- Compare old/new resolved config, request shapes, thresholds/grade bands,
  min-samples, window/cohort boundaries and known-finding check IDs. Keep trace
  evidence for counter resets, disappearing nodes and observation failures.
- Compare serial/parallel expanded sets and retain the full logical contract
  inventory, not just the number of YAML definitions.
- Require a real BATCH-enqueued/no-Fetch scenario with separate resource-owner
  observations, and high-hit-cache shrink construction before measuring recovery.
  The existing status-suppressed Master-TTL case alone does not establish the
  C++ onflight/KV lifecycle. Do not substitute NON_BATCH without opening a stream.

Run the oracle tests with:

```bash
python3 -m unittest discover -s tests -p test_contract_acceptance.py -v
```
