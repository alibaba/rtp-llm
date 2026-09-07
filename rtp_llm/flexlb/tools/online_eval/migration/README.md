# Migration contract ledger

The document's 131 legacy contracts are all present. The current registry has
139 contracts: five additional elastic contracts, `kv_leader_saturation_spill`,
`master_debug_snapshot` and `normal_no_fetch_observation` account for the extra
eight. The reviewed 29 target families remain the planning structure; supplemental
contracts are assigned explicitly to existing families, never silently dropped.

- `baseline.json` freezes current ordered registrations, profiles, findings and
  source digests. `baseline_revision_e505ab602d.json` records the independently
  reviewed correction of exactly three old contracts from revision 7419 to e505,
  including every before/after row and the separately hashed shared helper.
  Two impossible NON_BATCH crash lanes per restart case are excluded (375 to
  371 selected instances); duplicate FINISHED requires a settled replay baseline. A digest is an audit anchor, not proof of YAML equivalence.
- `target_manifest.json` assigns all 139 contracts to the 29 planned families and
  records module ownership, candidate source files and counts, pending contract
  IDs, and static reviews anchored to named revisions. Explicit `dependencies`
  record required shared fixes; a review cannot be applied to the original
  program without those fixes. `candidate_complete`
  only means all assigned IDs have explicit candidate programs; it does not mean
  runtime equivalence, full-profile Java execution or permission to delete them.
  `candidate_scenario_ids` also records temporary pilot names separately from the
  final target family. Planned definitions are not executable placeholders.
- `coverage.yaml` currently retains every complete legacy callable. No contract is
  marked migrated merely because a pilot has a similar name or parses correctly.
- `runtime_evidence.json` anchors selected real Java results to source revisions,
  artifact hashes and cleanup observations. The three admission passes apply
  only to their named batch-window instances; the Master passes likewise cover only
  two named lifecycle instances. The earlier elastic P6 failure
  remains recorded as unresolved; selected passes are not a full execution census.

Run `python3 migration/audit_manifest.py --out PATH` from `online_eval` to verify
bookkeeping against the actual registry and compiled YAML inventory. The audit
rejects omitted contracts, dropped profiles, duplicate family assignments and
unreviewed source drift. It also compares candidate IDs, remaining IDs, counts,
source files and implementation status to compiled programs, and checks that
static-review metadata names only present candidates and uses full commit-hash
syntax. This syntax check does not verify that the Git object exists, bind a
reviewer signature, or compare reviewed blobs with the current candidate source. It reuses the independent acceptance oracle rather than
changing that oracle to fit current results.

The selected legacy inventory has 371 profile instances (122/86/91/72). These are
selection counts, not passed runs. The report separately lists runnable YAML
scenario, variant, instance and declared-check counts. `legacy.contract` is only
the existing callable's result boundary for retained coverage; it is not a claim
that each old function contains one assertion. Original assertions still need
stage/check decomposition and paired configuration, threshold, sample-window and
execution evidence before replacement.

Current ownership: admission, master and priority preemption belong to agent4;
balance and priority queue to agent6; cancellation, status and engine recovery to agent5; elastic to agent1;
KV capacity to agent4; other KV families and RPC faults to agent2. Core catalog, compiler and this ledger are integrated by agent2
from the owners' fixed commits. The shared checkout is not used for these edits.
The inventory includes core execution fixtures and pilots; its runnable-definition
count must not be presented as the number of completed target families.

To migrate a contract, explicitly replace its coverage entry with `disposition:
migrate` and concrete `targets` containing `instance_id`, `check_ids` and the
reviewed preserved `contract_digest`. Every compatible profile and finding must
remain mapped. Static bookkeeping is necessary but cannot authorize removing a
legacy function without successful paired execution and cleanup evidence.
