# Cancellation contract mapping

Source baseline: `7120c1ff19446f694cbd99c5c9c5545bbd11e0c2`; shared status fixes through `3a79155e4c`, core effective-axis/preemption support `eba715e911`. Old callable files remain the comparison source, never an execution adapter.

| Old case | Original predicates and ordering retained |
|---|---|
| [cancel_basic](../../flexlb_ft/cases/cancel/cancel_basic.py) | First output; Master Cancel plus direct Prefill Cancel only NON_BATCH; stream ends in 5 s; recovery. BATCH engine receipt <=5 s from Master RPC issuance and Master drain 10 s. |
| [cancel_idempotent](../../flexlb_ft/cases/cancel/cancel_idempotent.py) | Two separate Cancel calls; BATCH first engine receipt <=5 s and forward delta >=1; after second Cancel and recovery, forwarding delta ==0. Old closing drain result was not in passed: observation only. |
| [cancel_sibling_isolation](../../flexlb_ft/cases/cancel/cancel_sibling_isolation.py) | A/C output10, B output500, concurrent Schedule; first output only A/C. Cancel B; B ends5; BATCH scheduler<=2 in2 s; A/C end30 and completed; B not completed and engine cancelled; recovery; BATCH drain10. |
| [cancel_after_terminal](../../flexlb_ft/cases/cancel/cancel_after_terminal.py) | Output1 completes before Cancel; worker RPC only NON_BATCH; no engine cancellation rewrite; recovery; BATCH drain10. Extra old 2 s wait was ignored, remains observation. |
| [cancel_unknown_rid](../../flexlb_ft/cases/cancel/cancel_unknown_rid.py) | Never-dispatched ID; Master reports found=false or typed NOT_FOUND; ledger fingerprint unchanged. No engine RPC. |
| [cancel_phase_timing](../../flexlb_ft/cases/cancel/cancel_phase_timing.py) | A has no first output after0.1 s client-only wait, B has first output within15 s. Each Cancel then stream-end5; BATCH per-request engine receipt5 anchored to RPC issuance. Recovery; BATCH Master drain95. |
| [cancel_anomaly_path](../../flexlb_ft/cases/cancel/cancel_anomaly_path.py) | First output precondition, Cancel, stream-end5, recovery, BATCH drain95. No extra engine receipt or NON_BATCH drain gate. |
| [cancel_deadline_exempt_inflight](../../flexlb_ft/cases/cancel/cancel_deadline_exempt_inflight.py) | Queue timeout2000 ms with enqueue_delay3000 ms; output500 completes without error within45 s; no engine cancellation; BATCH drain10; recovery while injection still installed, then clear. NON_BATCH fault remains original explicit no-op. |
| [cancel_schedule_drop_delivered](../../flexlb_ft/cases/cancel/cancel_schedule_drop_delivered.py) | enqueue_delay2000 ms; output500 Schedule.future begins asynchronously; wait0.5 s; cancel that client future (never Fetch); prove owned FutureCancelledError and worker exit within5 s. Cancel RPC delta>=1 within15 s; Master drain15; recovery before clearing. cancelled_rids is observation, not gate. |
| [cancel_preemption_victim](../../flexlb_ft/cases/cancel/cancel_preemption_victim.py) | 1P1D; priority ordering/default50, all three victim stages and engineCancellation50/1000 ms; decode capacity1. Unique keys, victimP30/input512/output200 reaches decode RUNNING10; concurrent P70/output2 arrives. Victim ends20 with exact8429 or proto2, engine cancelled, forward delta>=1; high succeeds30; BATCH Master drain15, engines clean15 and recovery. |
| [cancel_stream_break_prefill_autonomous](../../flexlb_ft/cases/cancel/cancel_stream_break_prefill_autonomous.py) | BATCH Fetch stream, output500/first output; only client transport cancel. Engine cancellation within10; engine clean15 and all three Master owners drain15; recovery. |
| [cancel_stream_break_decode_autonomous](../../flexlb_ft/cases/cancel/cancel_stream_break_decode_autonomous.py) | NON_BATCH Generate stream, output500/first output; only client transport cancel. Engine cancellation within10; engine clean15 and all three Master owners drain15; recovery. |
| [cancel_engine_notfound_settle](../../flexlb_ft/cases/cancel/cancel_engine_notfound_settle.py) | Output1 completed30; ordinary Cancel including NON_BATCH worker; an additional direct original-Prefill Cancel ACK has exact CANCEL_STATUS_NOT_FOUND; engine terminal not rewritten. BATCH drain10; recovery. |
| [cancel_engine_restarted_tombstoned_settle](../../flexlb_ft/cases/cancel/cancel_engine_restarted_tombstoned_settle.py) | 1P1D/output5000/first output; crash_after1 triggered by sacrificial Schedule(8 s), Master Prefill alive drops<=0 then restores>=1, each30 s, reconnect3 s. Master-only Cancel; client ends in5 s post-RPC wait, not business complete; Cancel forward delta>=1 within15. One direct same-RID Enqueue: no successes, exactly one same-RID8429. Engine clean45; scheduler residue<=1 within20 and non-growing after8; recovery. |
| [cancel_prefill_dead_await_terminal](../../flexlb_ft/cases/cancel/cancel_prefill_dead_await_terminal.py) | 1P1D/output500/first output; stop Prefill BEFORE Master-only Cancel. Client ends30 (completion or cancellation); three Master owners drain30; engines clean20; restore only stopped Prefill then recovery. |
| [cancel_decode_retire_closes_fence](../../flexlb_ft/cases/cancel/cancel_decode_retire_closes_fence.py) | 1P1D/output1000/first output; stop Prefill, Master-only Cancel, stop Decode. Client ends45 without business completion; Master drain45; engines clean30 BEFORE restore both and recovery. |
| [cancel_fencing_lost_on_engine_restart](../../flexlb_ft/cases/cancel/cancel_fencing_lost_on_engine_restart.py) | First crash/tombstone control as above; another crash wipes fence; wait original port readiness<=10, then exactly one direct same-RID Enqueue accepted>=1. Engine clean60; scheduler residue<=2 within20 and non-growing after8; recovery. No new assertion that accepted ACK has zero errors: old predicate only len(successes)>=1. |
| [cancel_transport_failure_one_shot](../../flexlb_ft/cases/cancel/cancel_transport_failure_one_shot.py) | 1P1D/output500/first output; cancel_no_respond before Master-only Cancel; end30 then wait2 and cancel counter delta EXACTLY1. Master drain30/engine clean20; clear fault before recovery. |
| [cancel_unexpected_status_await_terminal](../../flexlb_ft/cases/cancel/cancel_unexpected_status_await_terminal.py) | Same one-shot order with cancel_unexpected_status; end30, post-settle2 s counter delta EXACTLY1; Master drain30/engine clean20; clear then recovery. No direct worker Cancel under NON_BATCH. |

## Explicit representation differences

- Two crash_after cases require `enqueue_batch`: Java crash_after is implemented by EnqueueBatch, so NON_BATCH selections would not exercise the stated fault. This removes four invalid old profile selections, yielding 66 executable instances rather than 70. No 45-second bound was changed to 60; fence-lost already used 60 in old source.
- Old restart `settled_fast` is the return of a five-second wait after Cancel returns; the printed issuance-to-return latency was not in `passed`. The new hard check preserves that post-RPC wait. Basic/idempotent/phase receipt bounds do include issuance latency, exactly as their old helper.
- Unknown ID is allocated and never dispatched, replacing the old sentinel. Fingerprint uses the shared complete current owner schema, retaining raw Prefill members and Decode permit/load fields rather than fabricating zero for absent legacy Decode fields.
- Every terminating stream must provide real transport status, consumer done signal and independently verified thread exit. Intentional Schedule cancellation uses Python grpc.FutureCancelledError with the actual call.cancelled() proof, not an invented gRPC status. Unknown Python errors remain ERROR. HA stopped-Prefill programs explicitly permit UNAVAILABLE.
- Engine cancellation preserves the old disjunction: cancelled_rids contains RID OR request_lifecycle[RID].end_state is cancelled. Missing required snapshot fields are ERROR.
- Generic Master cleanup retains ONLY scheduler, Prefill batch count and Decode total load. Prefill membership is recorded but is not promoted into a cleanup assertion. Engine inflight and leak are separate hard checks.
- The crash trigger accepts only declared transport failures with complete Schedule exit evidence; its business outcome is not asserted. Independent health-drop and restore checks prove the fault effect. Cleanup owns requests, injection handles and fresh environment lifetime, including failure paths.
- Old docstring predictions do not become expected-fail classifications. All old passing predicates remain ordinary checks; old ignored observations stay observations.

## Executable check index

### cancel_fence_settlement / engine_notfound_settle_batch

`completed_before_cancel`, `terminal_ack_is_not_found`, `terminal_not_rewritten`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `recovery_succeeds`

### cancel_fence_settlement / engine_notfound_settle_nonbatch

`completed_before_cancel`, `terminal_ack_is_not_found`, `terminal_not_rewritten`, `recovery_succeeds`

### cancel_fence_settlement / prefill_dead_await_terminal

`first_output_received`, `terminal_ended`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `engine_drain_inflight`, `engine_drain_leaks`, `recovery_succeeds`

### cancel_fence_settlement / decode_retire_closes_fence

`first_output_received`, `terminal_ended`, `retirement_does_not_complete_business`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `engine_drain_inflight`, `engine_drain_leaks`, `recovery_succeeds`

### cancel_fence_settlement / transport_failure_one_shot

`first_output_received`, `terminal_ended`, `cancel_is_one_shot`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `engine_drain_inflight`, `engine_drain_leaks`, `recovery_succeeds`

### cancel_fence_settlement / unexpected_status_await_terminal

`first_output_received`, `terminal_ended`, `cancel_is_one_shot`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `engine_drain_inflight`, `engine_drain_leaks`, `recovery_succeeds`

### cancel_fence_settlement / engine_restarted_tombstoned_settle

`first_output_received`, `first_crash_health_dropped`, `first_crash_health_restored`, `settled_in_five_second_wait`, `cancel_did_not_complete_business`, `cancel_reached_fresh_engine`, `armed_fence_rejects_exact_rid_8429`, `engine_drain_inflight`, `engine_drain_leaks`, `residue_within_crash_trigger_bound`, `residue_does_not_grow`, `recovery_succeeds`

### cancel_fence_settlement / fencing_lost_on_engine_restart

`first_output_received`, `first_crash_health_dropped`, `first_crash_health_restored`, `settled_in_five_second_wait`, `cancel_did_not_complete_business`, `cancel_reached_fresh_engine`, `armed_fence_rejects_exact_rid_8429`, `second_crash_health_dropped`, `second_crash_health_restored`, `memory_only_fence_is_lost`, `engine_drain_inflight`, `engine_drain_leaks`, `residue_within_crash_trigger_bound`, `residue_does_not_grow`, `recovery_succeeds`

### cancel_lifecycle / basic_batch

`first_output_before_cancel`, `first_master_rpc_ok`, `stream_terminated`, `engine_receives_cancel_within_five`, `recovery_succeeds`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`

### cancel_lifecycle / idempotent_batch

`first_output_before_cancel`, `first_master_rpc_ok`, `engine_receives_first_cancel`, `first_cancel_forwarded`, `second_master_rpc_ok`, `stream_terminated`, `recovery_succeeds`, `second_cancel_not_forwarded`

### cancel_lifecycle / after_terminal_batch

`completed_before_cancel`, `first_master_rpc_ok`, `recovery_succeeds`, `terminal_not_rewritten_as_cancelled`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`

### cancel_lifecycle / basic_nonbatch

`first_output_before_cancel`, `first_master_rpc_ok`, `first_worker_rpc_ok`, `stream_terminated`, `recovery_succeeds`

### cancel_lifecycle / idempotent_nonbatch

`first_output_before_cancel`, `first_master_rpc_ok`, `first_worker_rpc_ok`, `second_master_rpc_ok`, `second_worker_rpc_ok`, `stream_terminated`, `recovery_succeeds`

### cancel_lifecycle / after_terminal_nonbatch

`completed_before_cancel`, `first_master_rpc_ok`, `first_worker_rpc_ok`, `recovery_succeeds`, `terminal_not_rewritten_as_cancelled`

### cancel_lifecycle / unknown_rid

`clean_baseline_scheduler`, `clean_baseline_prefill_batches`, `clean_baseline_decode_load`, `unknown_is_typed_not_found`, `unknown_does_not_mutate_ledger`

### cancel_lifecycle / anomaly_path_batch

`first_output_received`, `termination_ended`, `recovery_succeeds`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`

### cancel_lifecycle / sibling_isolation_batch

`a_first_received`, `c_first_received`, `b_termination_ended`, `cancelled_slot_removed`, `a_completion_ended`, `a_completed`, `c_completion_ended`, `c_completed`, `b_not_completed`, `b_engine_cancelled`, `recovery_succeeds`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`

### cancel_lifecycle / phase_timing_batch

`a_still_prefill`, `a_termination_ended`, `a_engine_receipt_within_five`, `b_first_received`, `b_termination_ended`, `b_engine_receipt_within_five`, `recovery_succeeds`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`

### cancel_lifecycle / deadline_exempt_inflight_batch

`completion_ended`, `inflight_exempt_completes`, `deadline_did_not_cancel_engine`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`, `recovery_succeeds`

### cancel_lifecycle / anomaly_path_nonbatch

`first_output_received`, `termination_ended`, `recovery_succeeds`

### cancel_lifecycle / sibling_isolation_nonbatch

`a_first_received`, `c_first_received`, `b_termination_ended`, `a_completion_ended`, `a_completed`, `c_completion_ended`, `c_completed`, `b_not_completed`, `b_engine_cancelled`, `recovery_succeeds`

### cancel_lifecycle / phase_timing_nonbatch

`a_still_prefill`, `a_termination_ended`, `b_first_received`, `b_termination_ended`, `recovery_succeeds`

### cancel_lifecycle / deadline_exempt_inflight_nonbatch

`completion_ended`, `inflight_exempt_completes`, `deadline_did_not_cancel_engine`, `recovery_succeeds`

### cancel_lifecycle / schedule_drop_delivered

`owned_schedule_future_cancelled`, `cancel_forwarded`, `closing_drain_scheduler`, `closing_drain_prefill_batches`, `closing_drain_decode_load`, `recovery_succeeds`

### cancel_lifecycle / stream_break_prefill_autonomous

`first_output_received`, `engine_observed_stream_cancellation`, `engine_drain_inflight`, `engine_drain_leaks`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `recovery_succeeds`

### cancel_lifecycle / stream_break_decode_autonomous

`first_output_received`, `engine_observed_stream_cancellation`, `engine_drain_inflight`, `engine_drain_leaks`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `recovery_succeeds`

### cancel_lifecycle / preemption_victim_batch

`victim_engine_owned`, `victim_terminal_ended`, `victim_typed_engine_preemption`, `high_completion_ended`, `high_completed`, `victim_engine_cancelled`, `weak_cancel_was_forwarded`, `master_drain_scheduler`, `master_drain_prefill_batches`, `master_drain_decode_load`, `engine_drain_inflight`, `engine_drain_leaks`, `recovery_succeeds`

### cancel_lifecycle / preemption_victim_nonbatch

`victim_engine_owned`, `victim_terminal_ended`, `victim_typed_engine_preemption`, `high_completion_ended`, `high_completed`, `victim_engine_cancelled`, `weak_cancel_was_forwarded`, `engine_drain_inflight`, `engine_drain_leaks`, `recovery_succeeds`

