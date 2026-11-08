import ast,json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from distributed_reclaim import reclaim_world,post_advice_sample_ready
from test_distributed_admission import FakeDist

class ReclaimWorldTest(unittest.TestCase):
    def _two_gate_cursor_exercise(self,*,fail_second=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'cpu-monitor').mkdir()
            (root/'cpu-monitor/CURRENT.json').write_text(json.dumps(dict(same_time_namespace=True,
                time_namespace='time:[1]',sample_started_monotonic=time.monotonic()+100,
                sample_completed_monotonic=time.monotonic()+100)))
            dist=FakeDist(4);rows=[dict(rank=i,pid=100+i,hostname='host',container_id='cid') for i in range(4)]
            instances=[];calls=[];counts={};lock=threading.Lock();results=[]
            class Cursor:
                def __init__(self,plan,*,pinned):
                    self.pinned=pinned;self.closed=False;self.directory=root/'cpu-reclaim';self.directory.mkdir();self.round=0;self.total=0;instances.append(self)
                def close(self):self.closed=True;self.pinned.close()
                def next_batch(self):
                    calls.append(dist.local.rank)
                    if fail_second and self.round==1:raise OSError('failure in second admission')
                    self.round+=1;self.total+=8192
                    out=dict(stage='exact_completed_source_ranges',round=self.round,advised_bytes=4096,
                        syscall_advised_bytes=8192,total_advised_bytes=self.total,wide_scanned_bytes=0)
                    (self.directory/('batch-%05d.json'%self.round)).write_text(json.dumps(out));return out
            def decision(*args,**kwargs):
                rank=dist.local.rank
                with lock:counts[rank]=counts.get(rank,0)+1;n=counts[rank]
                return dict(passed=(n%2==0))
            def worker(rank):
                dist.local.rank=rank;holder={}
                try:
                    for phase in ('first_adam','next_backward'):
                        reclaim_world(dist,object(),rows[rank],rows,root=root,run_id='r',world=4,reserve=0,
                            action_group=object(),cursor_holder=holder,phase=phase)
                    results.append('pass')
                except Exception as exc:results.append(type(exc).__name__)
            with patch('distributed_reclaim.PinnedSources') as pinned,patch('distributed_reclaim.evaluate',side_effect=decision),patch('distributed_reclaim.build_plan',return_value={'fixture':True}),\
                 patch('distributed_reclaim.ReclaimCursor',Cursor),patch('os.readlink',return_value='time:[1]'):
                threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
                for t in threads:t.start()
                for t in threads:t.join(8)
            self.assertTrue(all(not t.is_alive() for t in threads));self.assertEqual(len(results),4)
            self.assertEqual(len(instances),1);self.assertEqual(calls,[0,0])
            self.assertEqual(instances[0].total,8192 if fail_second else 16384)
            self.assertEqual(pinned.call_count,1);self.assertIs(instances[0].pinned,pinned.return_value)
            self.assertEqual(instances[0].closed,fail_second)
            return results

    def test_same_leader_cursor_and_ledger_across_two_gates(self):
        self.assertEqual(self._two_gate_cursor_exercise(),['pass']*4)
    def test_second_gate_failure_denies_all_and_does_not_restart_claim(self):
        self.assertEqual(self._two_gate_cursor_exercise(fail_second=True),['MemoryError']*4)
    def test_sample_crossing_advice_end_is_not_fresh(self):
        sample=dict(same_time_namespace=True,time_namespace='time:[1]',
                    sample_started_monotonic=9,sample_completed_monotonic=11,sample_unix=1000)
        self.assertFalse(post_advice_sample_ready(sample,10,'time:[1]'))
        sample.update(sample_started_monotonic=10.1,sample_completed_monotonic=11)
        self.assertTrue(post_advice_sample_ready(sample,10,'time:[1]'))
        self.assertFalse(post_advice_sample_ready(sample,10,'time:[2]'))
    def exercise(self,mode):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'cpu-monitor').mkdir();(root/'cpu-monitor/CURRENT.json').write_text(json.dumps({'sample_unix':time.time()+100,'same_time_namespace':True,'time_namespace':'time:[1]','sample_started_monotonic':time.monotonic()+100,'sample_completed_monotonic':time.monotonic()+100}))
            dist=FakeDist(4);rows=[dict(rank=i,pid=100+i,hostname='host',container_id='cid') for i in range(4)]
            result=[];counts={};lock=threading.Lock();leader_calls=[];regular_group=object();action_group=object()
            class Cursor:
                def __init__(self,plan,*,pinned):self.pinned=pinned;self.directory=root/'cpu-reclaim';self.directory.mkdir();self.round=0
                def close(self):self.pinned.close()
                def next_batch(self):
                    leader_calls.append(dist.local.rank)
                    if mode=='exhausted':return None
                    if mode=='read_failure':raise OSError('injected source read failure')
                    self.round+=1;r=dict(stage='exact_completed_source_ranges',round=self.round,advised_bytes=4096,
                        syscall_advised_bytes=8192,total_advised_bytes=8192,wide_scanned_bytes=0)
                    (self.directory/('batch-%05d.json'%self.round)).write_text(json.dumps(r));return r
            def decision(*args,**kwargs):
                rank=dist.local.rank
                with lock:counts[rank]=counts.get(rank,0)+1;round_=counts[rank]
                return dict(passed=(mode=='enough' or (mode=='one_batch' and round_>1)))
            def worker(rank):
                dist.local.rank=rank
                try:
                    out=reclaim_world(dist,regular_group,rows[rank],rows,root=root,run_id='r',world=4,reserve=0,action_group=action_group)
                    result.append((rank,'pass',out['rounds']))
                except Exception as exc:result.append((rank,type(exc).__name__,None))
            captured=[];original_gather=dist.all_gather_object
            def gather(out,value,group=None):
                self.assertIs(group,action_group if 'leader' in value else regular_group)
                if isinstance(value,dict) and value.get('leader'):captured.append(value.copy())
                return original_gather(out,value,group=group)
            dist.all_gather_object=gather
            with patch('distributed_reclaim.PinnedSources') as source,patch('distributed_reclaim.evaluate',side_effect=decision),patch('distributed_reclaim.build_plan',side_effect=ValueError('plan failed') if mode in ('plan_failure','plan_close_failure') else None,return_value={'fixture':True}),patch('distributed_reclaim.ReclaimCursor',Cursor),patch('os.readlink',return_value='time:[1]'):
                if mode=='plan_close_failure':source.return_value.close.side_effect=OSError('close failed')
                threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
                for t in threads:t.start()
                for t in threads:t.join(8)
            self.assertEqual(len(result),4);self.assertTrue(all(not t.is_alive() for t in threads))
            if mode in ('plan_failure','plan_close_failure'):source.return_value.close.assert_called_once_with()
            if mode in ('plan_failure','plan_close_failure'):
                failure=json.loads((root/'cpu-reclaim'/'ACTION-FAILED.json').read_text())
                self.assertEqual(failure['failure_type'],'ValueError');self.assertIsNone(failure['failure_message'])
                self.assertTrue(failure['failure_message_redacted']);self.assertNotIn('plan failed',json.dumps(failure))
                self.assertFalse(failure['partial_current_batch_may_have_executed'])
                self.assertEqual(failure['source_close_error_type'],'OSError' if mode=='plan_close_failure' else None)
            if mode=='plan_close_failure':
                self.assertEqual(captured[0]['error'],'ValueError: plan failed')
                self.assertEqual(captured[0]['source_close_error_type'],'OSError')
            return result,leader_calls
    def test_enough_never_creates_coordinator_or_advice(self):
        result,calls=self.exercise('enough');self.assertEqual(calls,[])
        self.assertTrue(all(state=='pass' and rounds==0 for _,state,rounds in result))
    def test_one_leader_one_batch_then_fresh_success(self):
        result,calls=self.exercise('one_batch');self.assertEqual(calls,[0])
        self.assertTrue(all(state=='pass' and rounds==1 for _,state,rounds in result))
    def test_exhausted_budget_denies_every_rank(self):
        result,calls=self.exercise('exhausted');self.assertEqual(calls,[0])
        self.assertTrue(all(state=='MemoryError' for _,state,_ in result))
    def test_source_read_failure_denies_every_rank(self):
        result,calls=self.exercise('read_failure');self.assertEqual(calls,[0])
        self.assertTrue(all(state=='MemoryError' for _,state,_ in result))
    def test_plan_failure_closes_pin_and_denies_every_rank(self):
        result,calls=self.exercise('plan_failure');self.assertEqual(calls,[])
        self.assertTrue(all(state=='MemoryError' for _,state,_ in result))
    def test_plan_and_close_both_fail_preserves_original_error(self):
        result,calls=self.exercise('plan_close_failure');self.assertEqual(calls,[])
        self.assertTrue(all(state=='MemoryError' for _,state,_ in result))
    def test_action_group_must_be_distinct_complete_gloo(self):
        class Dist:
            def __init__(self,backend='gloo',size=4):self.backend=backend;self.size=size
            def get_backend(self,group):return self.backend
            def get_world_size(self,group):return self.size
        regular=object()
        for d,action in [(Dist(),None),(Dist(),regular),(Dist('nccl'),object()),(Dist('gloo',3),object())]:
            with self.subTest(backend=d.backend,size=d.size,action=action):
                with self.assertRaisesRegex(ValueError,'action'):
                    reclaim_world(d,regular,{},[],root='/',run_id='r',world=4,reserve=0,action_group=action)
    def test_action_timeout_is_explicit_bounded_ascii_integer(self):
        tree=ast.parse((Path(__file__).parent/'qwen38_bf16_training_runtime.py').read_text())
        fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_reclaim_action_timeout_seconds')
        namespace={};exec(compile(ast.Module(body=[fun],type_ignores=[]),'timeout-contract','exec'),namespace)
        parse=namespace[fun.name]
        self.assertEqual(parse('600'),600)
        for bad in (None,'','120','3601','-1','6.0','６００',' 600'):
            with self.subTest(value=bad),self.assertRaises(ValueError):parse(bad)

if __name__=='__main__':unittest.main()
