import json,os,socket,tempfile,threading,time,unittest
import distributed_admission
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cpu_admission import GIB
from distributed_admission import first_adam_admission

class FakeDist:
    def __init__(self,world):
        self.world=world;self.local=threading.local();self.barrier=threading.Barrier(world,timeout=5)
        self.lock=threading.Lock();self.round=0;self.rows={}
    def is_initialized(self):return True
    def get_world_size(self):return self.world
    def get_rank(self):return self.local.rank
    def get_backend(self,group):return 'gloo'
    def all_gather_object(self,out,value,group=None):
        with self.lock:self.rows.setdefault(self.round,{})[self.local.rank]=value
        self.barrier.wait()
        with self.lock:out[:]=[self.rows[self.round][i] for i in range(self.world)]
        token=self.barrier.wait()
        if token==0:
            with self.lock:self.round+=1
        self.barrier.wait()

class AdmissionConcurrencyTest(unittest.TestCase):
    def run_world(self,*,deny=False,broken_audit=False,final_one_rank_denies=False,old_before_reclaim=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'cpu-monitor').mkdir()
            stats=dict(anon=100*GIB,file=0,inactive_file=0,file_dirty=0,file_writeback=0,file_mapped=0,shmem=0,unevictable=0)
            sample=dict(run_id='run',hostname='fake-host',container_id='c'*64,identity_verified=True,
                same_time_namespace=True,time_namespace='time:[1]',sample_started_monotonic=time.monotonic()-.1,
                sample_completed_monotonic=time.monotonic(),sample_unix=time.time(),cgroups=[dict(path='/c',is_exact_container=True,identity_verified=True,
                    limit_bytes=1000*GIB,current_bytes=(990 if deny else 100)*GIB,stat=stats,
                    events_delta={'oom':0,'oom_kill':0},pressure='some avg10=0')])
            if old_before_reclaim:
                sample['sample_unix']-=30;sample['sample_started_monotonic']-=30;sample['sample_completed_monotonic']-=30
            (root/'cpu-monitor/CURRENT.json').write_text(json.dumps(sample))
            nodes=root/'nodes.json';nodes.write_text(json.dumps({'nodes':[dict(container_ip='10.0.0.1',gpu_container='c'*64)]}))
            audit=root/'audit'
            if broken_audit:audit.write_text('not a directory')
            env=dict(Q38_RUN_ID='run',Q38_RUN_ROOT=str(root),Q38_TRAINING_AUDIT_DIR=str(audit),Q38_NODES_FILE=str(nodes),Q38_CPU_CACHE_RECLAIM_ENABLED='0')
            if old_before_reclaim:env['Q38_CPU_CACHE_RECLAIM_ENABLED']='1'
            dist=FakeDist(4);results=[]
            def worker(rank):
                dist.local.rank=rank
                engine=SimpleNamespace(_q38_planned_moment_bytes=10*GIB,
                    _q38_gradient_store=SimpleNamespace(cpu_spilled_bytes=GIB),
                    optimizer=SimpleNamespace(moments=SimpleNamespace(bytes=0)))
                try:first_adam_admission(engine,dist,object(),world=4);results.append((rank,'admitted'))
                except Exception as exc:results.append((rank,type(exc).__name__))
            original=distributed_admission.evaluate;calls={};lock=threading.Lock()
            def evaluate_override(*args,**kwargs):
                result=original(*args,**kwargs);rank=dist.local.rank
                with lock:calls[rank]=calls.get(rank,0)+1;number=calls[rank]
                if final_one_rank_denies and rank==2 and number==2:result['passed']=False
                return result
            def simulate_long_reclaim(*args,**kwargs):
                # The pre-reclaim fixture is 30s old; publish a different
                # fresh sample before returning, without a real 30s sleep.
                fresh=dict(sample,sample_unix=time.time(),sample_started_monotonic=time.monotonic()-.01,
                           sample_completed_monotonic=time.monotonic())
                temp=root/'cpu-monitor'/('CURRENT-%d.tmp'%dist.local.rank)
                temp.write_text(json.dumps(fresh));os.replace(temp,root/'cpu-monitor/CURRENT.json')
                return {'passed':True}
            with patch.dict(os.environ,env),patch.object(socket,'gethostname',return_value='fake-host'),patch.object(socket,'gethostbyname',return_value='10.0.0.1'),patch.object(os,'readlink',return_value='time:[1]'),patch.object(distributed_admission,'evaluate',side_effect=evaluate_override),patch('distributed_reclaim.reclaim_world',side_effect=simulate_long_reclaim):
                threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
                for t in threads:t.start()
                for t in threads:t.join(8)
            self.assertTrue(all(not t.is_alive() for t in threads));self.assertEqual(len(results),4)
            return sorted(v for _,v in results)
    def test_all_pass_only_after_unanimous_vote_and_audit(self):
        self.assertEqual(self.run_world(),['admitted']*4)
    def test_insufficient_memory_denies_every_rank(self):
        self.assertEqual(self.run_world(deny=True),['MemoryError']*4)
    def test_failed_audit_denies_every_rank(self):
        self.assertEqual(self.run_world(broken_audit=True),['MemoryError']*4)
    def test_final_single_rank_recheck_denies_every_rank(self):
        self.assertEqual(self.run_world(final_one_rank_denies=True),['MemoryError']*4)
    def test_old_pre_reclaim_sample_is_not_reused(self):
        self.assertEqual(self.run_world(old_before_reclaim=True),['admitted']*4)

if __name__=='__main__':unittest.main()
