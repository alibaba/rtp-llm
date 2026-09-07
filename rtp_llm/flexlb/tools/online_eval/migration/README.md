# Migration contract ledger

The document's 131 legacy contracts are all present. The current registry has
139 contracts: five additional elastic contracts, `kv_leader_saturation_spill`,
`master_debug_snapshot` and `normal_no_fetch_observation` account for the extra
eight. The reviewed 29 target families remain the planning structure; supplemental
contracts are assigned explicitly to existing families, never silently dropped.

- `baseline.json` freezes current ordered registrations, profiles, findings and
  source digests. A digest is an audit anchor, not proof of YAML equivalence.
- `target_manifest.json` assigns all 139 contracts to the 29 planned families and
  records module ownership. Planned definitions are not executable placeholders.
- `coverage.yaml` currently retains every complete legacy callable. No contract is
  marked migrated merely because a pilot has a similar name or parses correctly.

Run `python3 migration/audit_manifest.py --out PATH` from `online_eval` to verify
bookkeeping against the actual registry and compiled YAML inventory. The audit
rejects omitted contracts, dropped profiles, duplicate family assignments and
unreviewed source drift. It reuses the independent acceptance oracle rather than
changing that oracle to fit current results.

The selected legacy inventory has 375 profile instances (122/88/91/74). These are
selection counts, not passed runs. The report separately lists runnable YAML
scenario, variant, instance and declared-check counts. `legacy.contract` is only
the existing callable's result boundary for retained coverage; it is not a claim
that each old function contains one assertion. Original assertions still need
stage/check decomposition and paired configuration, threshold, sample-window and
execution evidence before replacement.

To migrate a contract, explicitly replace its coverage entry with `disposition:
migrate` and concrete `targets` containing `instance_id`, `check_ids` and the
reviewed preserved `contract_digest`. Every compatible profile and finding must
remain mapped. Static bookkeeping is necessary but cannot authorize removing a
legacy function without successful paired execution and cleanup evidence.
