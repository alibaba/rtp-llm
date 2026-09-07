# Status contract mapping

Source baseline: `9821d9dc73e0ded021ebf85ebfe1fe26af5f625f`; the 25 legacy status source files are unchanged in this worktree. Two additional mappings come from `master_debug_snapshot` and `normal_no_fetch_observation`. This is declaration coverage, not a claim that all variants or profiles passed remotely.

Every name below denotes `<stage>.contract` unless it is explicitly described as an observation or required source. A `*_drained_*`, `after_*_*`, or `clean_baseline_*` group expands into the separately listed scheduler, Prefill batch, Prefill member and Decode total-load checks. Each variant also maps old `master_ok` to `master_http_200.contract` and has explicit teardown plus mandatory owner cleanup.

The runner continues after ordinary failed checks to collect independent results. An action error blocks dependent stages; missing evidence is never converted to a finding.

## batch_ack_and_execution / ack_partial

`status_ack_partial_fail` — profiles: batch-window.

hang_free → failed_member_released_promptly; drained_a → ledger_drained_*; transient_ok → transient_policy; permanent_fast → permanent_member_isolation; permanent_no_retry → permanent_dispatch_nonempty + permanent_no_retry; inflight_ok → final_drained_*.

Checks: `failed_member_released_promptly.contract`, `ledger_drained_scheduler.contract`, `ledger_drained_prefill_batches.contract`, `ledger_drained_prefill_members.contract`, `ledger_drained_decode_load.contract`, `transient_policy.contract`, `permanent_member_isolation.contract`, `permanent_member_isolation_phase.contract`, `permanent_dispatch_nonempty.contract`, `permanent_no_retry.contract`, `final_drained_scheduler.contract`, `final_drained_prefill_batches.contract`, `final_drained_prefill_members.contract`, `final_drained_decode_load.contract`, `master_http_200.contract`.

Finding checks: `transient_policy.contract`.

## batch_ack_and_execution / execution_partial

`status_batch_async_partial_fail` — profiles: batch-window.

typed_terminal → normal_execution_terminal; inflight_ok → normal_drained_*; stable → no_resurrection; recovery_ok → healthy_recovery_success; typed_terminal_2 → serial_execution_terminal; inflight_ok2 → serial_drained_*.

Checks: `normal_execution_terminal.contract`, `normal_execution_terminal_phase.contract`, `normal_drained_scheduler.contract`, `normal_drained_prefill_batches.contract`, `normal_drained_prefill_members.contract`, `normal_drained_decode_load.contract`, `no_resurrection.contract`, `healthy_recovery_success.contract`, `serial_execution_terminal.contract`, `serial_execution_terminal_phase.contract`, `serial_drained_scheduler.contract`, `serial_drained_prefill_batches.contract`, `serial_drained_prefill_members.contract`, `serial_drained_decode_load.contract`, `master_http_200.contract`.

## batch_ack_and_execution / ack_multi_error

`status_ack_multi_error` — profiles: batch-window.

passthrough → code_8431_passthrough + code_8510_passthrough; no_resurrect → no_resurrection (initial scheduler drain remains an explicit precondition).

Checks: `code_8431_passthrough.contract`, `code_8431_passthrough_phase.contract`, `code_8510_passthrough.contract`, `code_8510_passthrough_phase.contract`, `failed_batches_drained_scheduler.contract`, `failed_batches_drained_prefill_batches.contract`, `failed_batches_drained_prefill_members.contract`, `failed_batches_drained_decode_load.contract`, `no_resurrection.contract`, `master_http_200.contract`.

## batch_ack_and_execution / ack_drop

`status_ack_empty_no_crash` — profiles: batch-window.

residue_ok → fence_residue_bounded + fence_residue_non_growing; drained/final==0 → quarantine_eventually_drains; request fate stays observational after a bounded wait.

Checks: `fence_residue_bounded.contract`, `fence_residue_non_growing.contract`, `quarantine_eventually_drains.contract`, `master_http_200.contract`.

Finding checks: `quarantine_eventually_drains.contract`.

## status_protocol / inflight_ttl_cleanup

`status_inflight_ttl_cleanup` — profiles: batch-window.

accepted_ok precondition → six_prefill_accepted; inflight_held>0 → ledger_nonempty_after_twelve; cleanup_ok/final==0 → scheduler_ttl_drain; ttl_events_ok → six_ttl_eviction_events; recovery_ok → recovery_success.

Checks: `six_prefill_accepted.contract`, `ledger_nonempty_after_twelve.contract`, `scheduler_ttl_drain.contract`, `six_ttl_eviction_events.contract`, `recovery_success.contract`, `master_http_200.contract`.

## status_protocol / prefill_suppress_all

`status_prefill_suppress_all` — profiles: batch-window.

legal_terminal → legal_request_terminals; sched_zero → scheduler_retires; batches_zero → prefill_retires; final_sched/final_batches → final_scheduler_zero + final_prefill_batches_zero; ttl_channel_ok/sched_channel_ok → prefill_ttl_channel_reachable + scheduler_ttl_channel_reachable; recovery_ok → recovery_success.

Checks: `legal_request_terminals.contract`, `scheduler_retires.contract`, `prefill_retires.contract`, `scheduler_ttl_channel_reachable.contract`, `prefill_ttl_channel_reachable.contract`, `final_scheduler_zero.contract`, `final_prefill_batches_zero.contract`, `recovery_success.contract`, `master_http_200.contract`.

## status_protocol / prefill_suppress_finished

`status_prefill_suppress_finished` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

legal_terminal → legal_request_terminals; inflight_ok → after_clear_*.

Checks: `legal_request_terminals.contract`, `after_clear_scheduler.contract`, `after_clear_prefill_batches.contract`, `after_clear_prefill_members.contract`, `after_clear_decode_load.contract`, `master_http_200.contract`.

## status_protocol / decode_suppress_finished

`status_decode_suppress_finished` — profiles: batch-window.

legal_terminal → legal_request_terminals; p_batches_zero → prefill_finishes_independently; d_requests_zero/final_d==0 → decode_retires_after_clear.

Checks: `legal_request_terminals.contract`, `prefill_finishes_independently.contract`, `decode_retires_after_clear.contract`, `master_http_200.contract`.

## status_protocol / no_respond

`status_status_no_respond` — profiles: batch-window.

alive_dropped → at_least_one_prefill_retired; all_retired → all_prefill_retired; drained → retired_ledger_drained; final_sched==0 → final_scheduler_zero; ttl_channel_ok → scheduler_ttl_channel_reachable; alive_back → topology_recovers; recovery_ok → recovery_success.

Checks: `at_least_one_prefill_retired.contract`, `all_prefill_retired.contract`, `retired_ledger_drained.contract`, `scheduler_ttl_channel_reachable.contract`, `final_scheduler_zero.contract`, `topology_recovers.contract`, `recovery_success.contract`, `master_http_200.contract`.

## status_protocol / version_regress

`status_version_regress` — profiles: batch-window.

alive_dropped → at_least_one_prefill_retired; drained → retired_ledger_drained; final_sched==0 → final_scheduler_zero; ttl_channel_ok → scheduler_ttl_channel_reachable; topology recovery remains observation.

Checks: `at_least_one_prefill_retired.contract`, `retired_ledger_drained.contract`, `scheduler_ttl_channel_reachable.contract`, `final_scheduler_zero.contract`, `master_http_200.contract`.

## status_protocol / unknown_rid_finished

`status_unknown_rid_finished` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

clean0 → clean_baseline_*; unchanged → unknown_terminal_ignored.

Checks: `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `unknown_terminal_ignored.contract`, `master_http_200.contract`.

## status_protocol / unknown_rid_running

`status_unknown_rid_running` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

clean0 → clean_baseline_*; drained/final==0 → unknown_active_retires_after_clear; whether ACTIVE initially registers is observation.

Checks: `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `unknown_active_retires_after_clear.contract`, `master_http_200.contract`.

## status_protocol / unknown_batchid

`status_unknown_batchid` — profiles: batch-window.

control_err is None → baseline_request_success; target_err is None → real_request_unaffected; inflight_ok → after_clear_*.

Checks: `baseline_request_success.contract`, `real_request_unaffected.contract`, `after_clear_scheduler.contract`, `after_clear_prefill_batches.contract`, `after_clear_prefill_members.contract`, `after_clear_decode_load.contract`, `master_http_200.contract`.

## status_protocol / special_ids

`status_special_ids` — profiles: batch-window.

clean0 → clean_baseline_*; rid_neg_ignored → negative_rid_ignored; real_rid_unaffected → zero_batch_unaffected + negative_batch_unaffected; inflight_ok → after_clear_*; RID zero channel acceptance/refusal stays observation.

Checks: `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `negative_rid_ignored.contract`, `zero_batch_unaffected.contract`, `negative_batch_unaffected.contract`, `after_clear_scheduler.contract`, `after_clear_prefill_batches.contract`, `after_clear_prefill_members.contract`, `after_clear_decode_load.contract`, `master_http_200.contract`.

## status_protocol / unbatched_single_request

`status_unbatched_single_request` — profiles: batch-window.

clean0 → clean_baseline_*; all_noop → omitted_running_ignored + omitted_finished_ignored + zero_running_ignored + zero_finished_ignored; inflight_ok → after_clear_*.

Checks: `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `omitted_running_ignored.contract`, `omitted_finished_ignored.contract`, `zero_running_ignored.contract`, `zero_finished_ignored.contract`, `after_clear_scheduler.contract`, `after_clear_prefill_batches.contract`, `after_clear_prefill_members.contract`, `after_clear_decode_load.contract`, `master_http_200.contract`.

## status_protocol / foreign_batchid

`status_foreign_batchid` — profiles: batch-window.

clean0 → clean_baseline_*; ghost_ignored → foreign_terminal_ignored; real_unaffected → real_traffic_success; inflight_ok → after_foreign_*.

Checks: `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `foreign_terminal_ignored.contract`, `real_traffic_success.contract`, `after_foreign_scheduler.contract`, `after_foreign_prefill_batches.contract`, `after_foreign_prefill_members.contract`, `after_foreign_decode_load.contract`, `master_http_200.contract`.

## status_protocol / duplicate_finished

`status_duplicate_finished` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

ok==4 → traffic_success; clean_ok → after_clear_*; stable → terminal_replay_is_idempotent; scheduler drain before replay remains a precondition.

Checks: `traffic_success.contract`, `scheduler_retires.contract`, `terminal_replay_is_idempotent.contract`, `after_clear_scheduler.contract`, `after_clear_prefill_batches.contract`, `after_clear_prefill_members.contract`, `after_clear_decode_load.contract`, `master_http_200.contract`.

## status_protocol / cursor_regress

`status_cursor_regress` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

clean_ok → clean_baseline_*; stable → cursor_replay_is_idempotent; still_clean → after_replay_*; recovery_ok → recovery_success.

Checks: `history_success.contract`, `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `cursor_replay_is_idempotent.contract`, `after_replay_scheduler.contract`, `after_replay_prefill_batches.contract`, `after_replay_prefill_members.contract`, `after_replay_decode_load.contract`, `recovery_success.contract`, `master_http_200.contract`.

## status_protocol / finished_then_running

`status_finished_then_running` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

baseline request precondition → settled_success; clean_ok → clean_baseline_*; no_resurrect_during → terminal_cannot_resurrect; clean_final → after_replay_*.

Checks: `settled_success.contract`, `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `terminal_cannot_resurrect.contract`, `after_replay_scheduler.contract`, `after_replay_prefill_batches.contract`, `after_replay_prefill_members.contract`, `after_replay_decode_load.contract`, `master_http_200.contract`.

## status_protocol / zombie_completed_running

`status_zombie_completed_running` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

ok==4 → traffic_success; clean_ok → after_zombie_*; d_requests==0 → decode_load_zero; zombie counters remain raw mock observations.

Checks: `traffic_success.contract`, `after_zombie_scheduler.contract`, `after_zombie_prefill_batches.contract`, `after_zombie_prefill_members.contract`, `after_zombie_decode_load.contract`, `decode_load_zero.contract`, `master_http_200.contract`.

## status_protocol / zombie_fake_running

`status_zombie_fake_running` — profiles: batch-window, single-nonbatch, single-batch, window-nonbatch.

clean0 → clean_baseline_*; bounded → resident_growth_bounded; master_ok_during → master_healthy_during_active; drained/final==0 → ghosts_retire_after_clear; resident_after_2xTTL and peak are observations, not a zero-residency assertion.

Checks: `clean_baseline_scheduler.contract`, `clean_baseline_prefill_batches.contract`, `clean_baseline_prefill_members.contract`, `clean_baseline_decode_load.contract`, `resident_growth_bounded.contract`, `master_healthy_during_active.contract`, `ghosts_retire_after_clear.contract`, `master_http_200.contract`.

## status_protocol / decode_before_prefill

`status_decode_before_prefill` — profiles: batch-window.

ok==4 → decode_completes_requests; p_batches_fast → decode_terminal_retires_prefill_promptly; final_p==0 → prefill_eventually_retires; recovery_ok → recovery_success.

Checks: `decode_completes_requests.contract`, `decode_terminal_retires_prefill_promptly.contract`, `prefill_eventually_retires.contract`, `recovery_success.contract`, `master_http_200.contract`.

Finding checks: `decode_terminal_retires_prefill_promptly.contract`.

## status_protocol / decode_running_before_prefill

`status_decode_running_before_prefill` — profiles: batch-window.

dispatched → batch_really_dispatched; p_held → intermediate_cannot_retire_prefill; sched_held → intermediate_cannot_retire_scheduler; drained → after_clear_*.

Checks: `batch_really_dispatched.contract`, `intermediate_cannot_retire_prefill.contract`, `intermediate_cannot_retire_scheduler.contract`, `after_clear_scheduler.contract`, `after_clear_prefill_batches.contract`, `after_clear_prefill_members.contract`, `after_clear_decode_load.contract`, `master_http_200.contract`.

## status_protocol / decode_waiting_before_prefill

`status_decode_waiting_before_prefill` — profiles: batch-window.

dispatched → batch_really_dispatched; p_held → intermediate_cannot_retire_prefill; sched_held → intermediate_cannot_retire_scheduler; drained → after_clear_*.

Checks: `batch_really_dispatched.contract`, `intermediate_cannot_retire_prefill.contract`, `intermediate_cannot_retire_scheduler.contract`, `after_clear_scheduler.contract`, `after_clear_prefill_batches.contract`, `after_clear_prefill_members.contract`, `after_clear_decode_load.contract`, `master_http_200.contract`.

## status_protocol / fetch_error

`status_fetch_error` — profiles: batch-window.

surfaced → fetch_fault_surfaces; err2 is None → fresh_request_success; inflight_ok → after_fetch_error_*; engine_clean → prefill_engine_drained; recovery_ok → recovery_success.

Checks: `fetch_fault_surfaces.contract`, `fresh_request_success.contract`, `after_fetch_error_scheduler.contract`, `after_fetch_error_prefill_batches.contract`, `after_fetch_error_prefill_members.contract`, `after_fetch_error_decode_load.contract`, `prefill_engine_drained.contract`, `recovery_success.contract`, `master_http_200.contract`.

## status_protocol / debug_snapshot

`master_debug_snapshot` — profiles: batch-window.

completed member → completed_success; queryable resource-free scheduler tombstone → resource_free_queryable_tombstone; schema/generation/completeness failures remain ERROR.

Checks: `completed_success.contract`, `resource_free_queryable_tombstone.contract`, `master_http_200.contract`.

## status_protocol / normal_no_fetch

`normal_no_fetch_observation` — profiles: batch-window.

fresh generation → fresh_accepted_zero + fresh_fetch_zero; Schedule/enqueue → master_enqueued; nonempty completion → prefill_completed + one_prefill_acceptance; zero Fetch → completion_window_no_fetch + post_completion_no_fetch + client_never_fetches; post-completion hold → prefill_stays_completed; separate recovery → separate_recovery_success.

Checks: `fresh_accepted_zero.contract`, `fresh_fetch_zero.contract`, `prefill_completed.contract`, `master_enqueued.contract`, `completion_window_no_fetch.contract`, `prefill_stays_completed.contract`, `post_completion_no_fetch.contract`, `client_never_fetches.contract`, `one_prefill_acceptance.contract`, `separate_recovery_success.contract`, `master_http_200.contract`.

## Explicit semantic changes and observations

- New assertions ending in `_phase.contract` prove Schedule rejection versus post-Schedule execution error. They are separate from the old count/code predicates and are ordinary checks, not findings. The old string matcher alone did not establish the protocol phase.
- Only `transient_policy`, `quarantine_eventually_drains`, and `decode_terminal_retires_prefill_promptly` are findings. The old zombie whole-case `expected_fail` could swallow baseline, health or clear-after-drain regressions; migration explicitly removes that classification. No active-ghost zero-residency assertion is added.
- Zombie ACTIVE observation is 60 seconds at five-second intervals; clear-after-retirement remains 95 seconds. Continuously reported ACTIVE refreshes Java Master activity. Sixty seconds after injection is not sixty seconds of status silence. These are Java Master scheduler/RequestRegistry observations, not C++ deferred slots or KV ownership.
- The unknown-batch legacy injection used ignored `batchId`/`errorCode` aliases. Migration preserves its effective `batch_id=0` behavior and makes it explicit; it does not relabel this as a positive wrong-batch-ID test. `foreign_batchid` separately uses the effective snake-case 10000000 value. RID zero is an injection-channel probe only.
- Strict live Decode fields replace missing legacy `inflight_requests`/`inflight_batches` defaults in fingerprints. Missing owner schema now produces ERROR. Prefill acceptance is measured only on Prefill engines; it cannot be inflated by Decode acceptance.
- Suppression/transient legal terminals retain the old `_timeout_typed` vocabulary. A recorded request RPC deadline is allowed only where declared and after transport terminal, stream end, consumer done and independently verified exit. A stage deadline, cancelled consumer, untyped ERROR or missing evidence stays TIMEOUT/ERROR. Fetch-error explicitly permits gRPC UNKNOWN because the mock calls `onError(new RuntimeException("injected fetch_error"))`; it does not permit arbitrary Python exceptions.
- Metric channels are required and checked independently. Absent TTL series in a successful parsed exposition are sparse zero counters; unavailable HTTP or malformed owner fields are errors. Only `inflight_ttl_cleanup` asserts a positive scheduler-eviction delta; the other TTL deltas and old shared-log anchors remain observations. Shared log offsets are not used as required evidence.
- Debug/noFetch now require every captured debug sample to be complete, a stricter evidence rule than the old noFetch loop, which tolerated partial intermediate frames. The first YAML run encountered a partial snapshot and correctly reported ERROR; a later fresh run passed without relaxing this rule. This is not a claim of stable repeated success. Source errors preserve a raw debug artifact.
- Normal noFetch has a fresh backend instance, no injection, no wait/Fetch on its original cohort, a two-second post-completion observation and a different recovery request. Owner rows are observations, not a universal zero-owner assertion. No GPU, C++ onflight, connector KV or exact 600-second lifetime is tested.
