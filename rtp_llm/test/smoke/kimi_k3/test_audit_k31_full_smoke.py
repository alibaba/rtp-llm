"""A request result or Graph bucket is insufficient execution evidence."""
import copy
import unittest

from audit_k31_full_smoke import (IndexedEvents, audit, group_evidence, replay_evidence,
                                 host_cancel_evidence, large_batch_evidence)
from k31_full_smoke import cache_pressure_plan
from smoke_request_identity import resolve


class MtpExecutionAuditTest(unittest.TestCase):
    def test_bs63_requires_the_complete_32_31_cohort_at_one_step(self):
        ids = {f'case{i}': [100+i] for i in range(63)}
        stages = [dict(name='orthogonal_decode_03_batch_63', case_names=list(ids))]
        events = {rank: [] for rank in range(8)}
        for owner in (0, 1):
            events[owner*4] = [dict(event='mtp_target_verify_forward', fake=False,
                execution_step=7, request_ids=list(range(100+owner, 163, 2)))]
        proof = large_batch_evidence(stages, ids, events, 4)
        self.assertEqual(proof[0]['owner_batches'], [32, 31])
        events[4][0]['execution_step'] = 8
        with self.assertRaisesRegex(ValueError, 'same-step'):
            large_batch_evidence(stages, ids, events, 4)

    def test_http_identity_keeps_equivalent_candidates_and_rejects_extra_traffic(self):
        stages = [dict(name='pair', passed=True, case_names=['a', 'b'])]
        row = dict(input_len=9, prefill_total_reuse_len=0,
                   decode_owner_rank=0, iter_count=2)
        raw = {name: dict(passed=True, result=dict(row), transport='http') for name in ('a','b')}
        p = {rank: [] for rank in range(8)}
        p[0] = [dict(event='mtp_target_prefill_forward', fake=False, request_ids=[10,11],
                     context_lengths=[9,9], reuse_lengths=[0,0], execution_step=1)]
        d = {rank: [] for rank in range(8)}
        d[0] = [dict(event='mtp_target_verify_forward', fake=False, request_ids=[10,11])]
        candidates, methods, _ = resolve(stages, raw, p, d)
        self.assertEqual(candidates, {'a':[10,11], 'b':[10,11]})
        self.assertTrue(all('equivalent_signature_set' in m for m in methods.values()))
        p[0].append(dict(event='mtp_target_prefill_forward', fake=False, request_ids=[12],
                         context_lengths=[9], reuse_lengths=[0], execution_step=2))
        with self.assertRaisesRegex(AssertionError, 'unexplained native traffic'):
            resolve(stages, raw, p, d)

    def group(self):
        events = {}
        for rank in range(8):
            events[rank] = [dict(event="mtp_target_verify_forward", execution_step=9,
                world_rank=rank, dp_rank=rank//4, tp_rank=rank%4,
                logical_authoritative=rank%4==0, logical_batch=2 if rank%4==0 else -1,
                request_ids=[10,11] if rank%4==0 else [], fake=False,
                input_rows=2,token_rows=8,time_ns=100)]
        return events, events[4][0]

    def test_owner_root_and_all_its_tp_ranks_must_execute(self):
        events, root = self.group()
        self.assertEqual(group_evidence(events,root,4),[10,11])
        events[6].clear()
        with self.assertRaisesRegex(ValueError,"missing"):
            group_evidence(events,root,4)

    def test_synchronized_rows_and_physical_rank_are_checked(self):
        for field,value in (("token_rows",4),("world_rank",4),("dp_rank",0),("fake",True)):
            events,root=self.group();events[7][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                group_evidence(events,root,4)

    def test_fake_or_non_authoritative_root_cannot_prove_a_batch(self):
        events, root = self.group()
        root = copy.deepcopy(root);root["logical_authoritative"]=False
        with self.assertRaisesRegex(ValueError,"authoritative"):
            group_evidence(events,root,4)

    def test_graph_must_match_the_phase_inside_its_call_window(self):
        _,forward=self.group()
        replay=dict(event="cuda_graph_replay",role=2,real_batch=2,bucket=4,time_ns=101)
        next_call=dict(event="mtp_draft_update_forward",execution_step=10,time_ns=102)
        replay_evidence([forward,replay,next_call],forward)
        for change in ({"role":1},{"time_ns":103},{"bucket":1}):
            bad=dict(replay,**change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                replay_evidence([forward,next_call,bad],forward)

    def test_incomplete_or_target_only_result_cannot_pass(self):
        with self.assertRaisesRegex(ValueError,"smoke MTP"):
            audit(dict(passed=True,target_only=True,cases=[]),{}, {},8)

    def test_pressure_plan_must_match_native_full_pool_capacity(self):
        result = dict(reuse_unit_tokens=8192, cache_pressure_plan=cache_pressure_plan(216, 8192))
        with self.assertRaisesRegex(ValueError, 'does not match the runtime pool'):
            audit(result, {}, {}, 8, prefill_full_pages=217)

    def test_indexed_snapshot_preserves_duplicates_and_graph_window_checks(self):
        events, root = self.group()
        indexed = {rank: IndexedEvents(rows) for rank, rows in events.items()}
        self.assertEqual(group_evidence(indexed, root, 4), [10, 11])
        indexed[6] = IndexedEvents(events[6] * 2)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            group_evidence(indexed, root, 4)
        replay = dict(event="cuda_graph_replay", role=2, real_batch=2, bucket=4, time_ns=101)
        boundary = dict(event="mtp_draft_update_forward", execution_step=10, time_ns=102)
        replay_evidence(IndexedEvents([root, replay, boundary]), root)
        with self.assertRaisesRegex(ValueError, "missing"):
            replay_evidence(IndexedEvents([root, boundary, dict(replay, time_ns=103)]), root)

    def test_cancel_requires_native_pending_snapshot_not_just_rpc_ack(self):
        start = dict(event="host_cache_load_started",request_id=91,time_ns=100)
        ack = dict(event="prefill_priority_cancel_accepted",request_id=91,time_ns=101)
        with self.assertRaisesRegex(ValueError,"native cancellation"):
            host_cancel_evidence(IndexedEvents([start,ack]))
        pending = dict(event="host_cache_priority_cancel_requested_while_pending",
                       request_id=91,host_reuse_len=57344,time_ns=101)
        self.assertEqual(host_cancel_evidence(IndexedEvents([start,pending,ack])),pending)
        for changed in ({"request_id":92},{"host_reuse_len":0},{"time_ns":99}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                host_cancel_evidence(IndexedEvents([start,dict(pending,**changed)]))
        done = dict(event="host_cache_load_done",request_id=91,time_ns=101)
        with self.assertRaises(ValueError):
            host_cancel_evidence(IndexedEvents([start,done,pending]))


if __name__=="__main__":
    unittest.main()
