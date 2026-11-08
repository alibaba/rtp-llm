import json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from distributed_reclaim import reclaim_world,post_advice_sample_ready
from test_distributed_admission import FakeDist

class ReclaimWorldTest(unittest.TestCase):
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
            result=[];counts={};lock=threading.Lock();leader_calls=[]
            class Cursor:
                def __init__(self,plan):self.directory=root/'cpu-reclaim';self.directory.mkdir();self.round=0
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
                    out=reclaim_world(dist,object(),rows[rank],rows,root=root,run_id='r',world=4,reserve=0)
                    result.append((rank,'pass',out['rounds']))
                except Exception as exc:result.append((rank,type(exc).__name__,None))
            with patch('distributed_reclaim.evaluate',side_effect=decision),patch('distributed_reclaim.build_plan',return_value={'fixture':True}),patch('distributed_reclaim.ReclaimCursor',Cursor),patch('os.readlink',return_value='time:[1]'):
                threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
                for t in threads:t.start()
                for t in threads:t.join(8)
            self.assertEqual(len(result),4);self.assertTrue(all(not t.is_alive() for t in threads))
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

if __name__=='__main__':unittest.main()
