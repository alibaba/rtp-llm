import hashlib,inspect,json,os,random,socket,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from cpu_admission import evaluate,GIB
from next_backward_admission import future_spill_upper_bytes,next_batch_admission,install_batch_boundary
from test_distributed_admission import FakeDist


def snapshot(*,oom=0,current=600*GIB,anon=580*GIB):
    now=time.monotonic();stat=dict(anon=anon,file=20*GIB,inactive_file=10*GIB,file_dirty=0,
        file_writeback=0,file_mapped=0,shmem=0,unevictable=0)
    child=dict(path='/pod/container',is_exact_container=True,identity_verified=True,
        limit_bytes=800*GIB,current_bytes=current,stat=stat,events_delta={'oom':oom,'oom_kill':0},pressure='ok')
    parent=dict(child,path='/pod',is_exact_container=False,current_bytes=current+20*GIB,
        stat=dict(stat,anon=anon+20*GIB))
    return dict(run_id='run',hostname='host',container_id='c'*64,identity_verified=True,
        same_time_namespace=True,time_namespace='time:[1]',sample_started_monotonic=now-.01,
        sample_completed_monotonic=now,sample_unix=time.time(),cgroups=[child,parent])


def peers(*,allocated=100*GIB,pending=20*GIB):
    return [dict(rank=i,hostname='host',planned_moment_bytes=100*GIB,allocated_moment_bytes=allocated,
        cpu_gradient_bytes=0,next_cpu_gradient_upper_bytes=pending) for i in range(4)]


class ProjectedBackwardTest(unittest.TestCase):
    def decide(self,s,p):
        return evaluate(s,p,run_id='run',hostname='host',expected_cid='c'*64,
            expected_time_namespace='time:[1]',reserve=60*GIB,phase='next_backward')

    def test_moments_already_charged_once_and_spill_upper_added(self):
        out=self.decide(snapshot(),peers());child,parent=out['cgroups']
        self.assertEqual(child['existing_moment_bytes'],400*GIB)
        self.assertEqual(child['pending_moment_bytes'],0)
        self.assertEqual(child['pending_gradient_upper_bytes'],80*GIB)
        self.assertEqual(child['required_bytes'],680*GIB)
        self.assertEqual(parent['required_bytes'],700*GIB)
        self.assertTrue(out['passed'])

    def test_parent_limit_and_oom_events_deny(self):
        self.assertFalse(self.decide(snapshot(current=710*GIB,anon=690*GIB),peers())['passed'])
        with self.assertRaises(ValueError):self.decide(snapshot(oom=1),peers())

    def test_incomplete_moments_existing_gradients_or_stale_sample_fail(self):
        with self.assertRaises(ValueError):self.decide(snapshot(),peers(allocated=99*GIB))
        p=peers();p[0]['cpu_gradient_bytes']=1
        with self.assertRaises(ValueError):self.decide(snapshot(),p)
        s=snapshot();s['sample_started_monotonic']-=20;s['sample_unix']-=20
        with self.assertRaises(ValueError):self.decide(s,peers())
        with self.assertRaises(ValueError):evaluate(snapshot(),peers(),run_id='run',hostname='host',
            expected_cid='c'*64,expected_time_namespace='time:[1]',phase='next_backward',reuse_enabled=True)

    def test_greedy_spill_bound(self):
        self.assertEqual(future_spill_upper_bytes(100,256,64),208)
        with self.assertRaises(ValueError):future_spill_upper_bytes(2,4,12)

    def test_greedy_bound_covers_random_orders_and_sizes(self):
        rng=random.Random(20261108)
        for _ in range(1000):
            sizes=[4*rng.randint(0,32) for _ in range(rng.randint(1,50))]
            limit=4*rng.randint(0,100);bound=future_spill_upper_bytes(sum(sizes)//4,limit,max(sizes))
            for _ in range(4):
                rng.shuffle(sizes);retained=spilled=0
                for size in sizes:
                    if retained+size<=limit:retained+=size
                    else:spilled+=size
                self.assertLessEqual(spilled,bound)

    def test_source_pinned_batch_wrapper_runs_gate_and_poison_on_failure(self):
        events=[]
        class Engine:
            _q38_cpu_admission_group='gloo'
            _q38_batch_completion_group='gloo2'
            _q38_gradient_store=SimpleNamespace(finish_gpu_capture_contract=lambda:events.append(('finish',None)))
            def train_batch(self,data,loss_function):
                if data=='fail':raise ValueError('original batch failed')
                events.append(('train',data));return data
        source_hash=hashlib.sha256(inspect.getsource(Engine.train_batch).encode()).hexdigest()
        def poison(engine):events.append(('poison',None));engine.failed=True
        dist=FakeDist(1);dist.local.rank=0
        install_batch_boundary(Engine,dist,expected_train_batch_sha256=source_hash,on_failure=poison,is_target=lambda e:getattr(e,'target',True),world=1)
        engine=Engine()
        with patch('next_backward_admission.next_batch_admission',side_effect=lambda *a,**k:events.append(('gate',None))):
            self.assertEqual(engine.train_batch('first',None),'first')
            self.assertEqual(engine.train_batch('second',None),'second')
        self.assertEqual(events,[('gate',None),('train','first'),('finish',None),('gate',None),('train','second'),('finish',None)])
        with patch('next_backward_admission.next_batch_admission',side_effect=MemoryError('denied')):
            with self.assertRaises(MemoryError):engine.train_batch('denied',None)
        self.assertTrue(engine.failed);self.assertEqual(events[-1],('poison',None))
        failing=Engine()
        with patch('next_backward_admission.next_batch_admission',return_value=None):
            with self.assertRaises(ValueError):failing.train_batch('fail',None)
        self.assertTrue(failing.failed);self.assertEqual(events[-1],('poison',None))
        other=Engine();other.target=False
        with patch('next_backward_admission.next_batch_admission',side_effect=AssertionError('must bypass')):
            self.assertEqual(other.train_batch('other',None),'other')
            with self.assertRaises(ValueError):other.train_batch('fail',None)
        self.assertFalse(hasattr(other,'failed'))

    def test_one_original_or_finish_failure_poisons_all_four_before_return(self):
        for mode in ('original','finish'):
            with self.subTest(mode=mode):
                dist=FakeDist(4);results=[];engines=[]
                class Engine:
                    _q38_cpu_admission_group='gloo'
                    _q38_batch_completion_group='gloo2'
                    def __init__(self,rank):
                        self.rank=rank
                        def finish():
                            if mode=='finish' and rank==3:raise RuntimeError('finish failure')
                        self._q38_gradient_store=SimpleNamespace(finish_gpu_capture_contract=finish)
                    def train_batch(self,data,loss_function):
                        if mode=='original' and self.rank==3:raise RuntimeError('backward failure')
                        return 'would export'
                source_hash=hashlib.sha256(inspect.getsource(Engine.train_batch).encode()).hexdigest()
                def poison(engine):engine.failed=True
                install_batch_boundary(Engine,dist,expected_train_batch_sha256=source_hash,on_failure=poison,
                    is_target=lambda e:True,world=4)
                def worker(rank):
                    dist.local.rank=rank;engine=Engine(rank);engines.append(engine)
                    try:engine.train_batch('batch',None);results.append('returned')
                    except Exception:results.append('failed')
                with patch('next_backward_admission.next_batch_admission',return_value=None):
                    threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
                    for t in threads:t.start()
                    for t in threads:t.join(8)
                self.assertTrue(all(not t.is_alive() for t in threads))
                self.assertEqual(results,['failed']*4)
                self.assertTrue(all(e.failed and e._q38_batch_in_progress for e in engines))


class WorldGateTest(unittest.TestCase):
    def run_world(self,*,world=4,step_mismatch=False,deny=False,initial=False,local_failure=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'cpu-monitor').mkdir()
            s=snapshot(current=(710 if deny else 600)*GIB,anon=(690 if deny else 580)*GIB)
            (root/'cpu-monitor/CURRENT.json').write_text(json.dumps(s))
            env=dict(Q38_RUN_ID='run',Q38_RUN_ROOT=str(root),Q38_TRAINING_AUDIT_DIR=str(root/'audit'),
                Q38_CPU_RESERVE_BYTES_PER_NODE=str(60*GIB),Q38_CPU_CACHE_RECLAIM_ENABLED='0')
            dist=FakeDist(world);result=[]
            def worker(rank):
                dist.local.rank=rank;engine=SimpleNamespace(_q38_reclaim_cursor_holder={})
                try:next_batch_admission(engine,dist,object(),world=world);result.append('pass')
                except Exception as exc:result.append(type(exc).__name__)
            def collect(engine,dist,group,world,run_id,host):
                rank=dist.local.rank;node=rank//4
                if local_failure and rank==0:raise ValueError('injected local ownership failure')
                return dict(rank=rank,pid=1,hostname='host' if world==4 else 'host'+str(node),
                    container_id='c'*64 if world==4 else str(node),run_id=run_id,
                    optimizer_steps=[] if initial or (step_mismatch and rank==0) else [1],
                    planned_moment_bytes=100*GIB,allocated_moment_bytes=0 if initial or (step_mismatch and rank==0) else 100*GIB,cpu_gradient_bytes=0,
                    next_cpu_gradient_upper_bytes=20*GIB,error=None)
            with patch.dict(os.environ,env),patch.object(socket,'gethostname',return_value='host'),\
                 patch.object(os,'readlink',return_value='time:[1]'),\
                 patch('next_backward_admission._collect_local',side_effect=collect):
                threads=[threading.Thread(target=worker,args=(i,)) for i in range(world)]
                for t in threads:t.start()
                for t in threads:t.join(10)
            self.assertTrue(all(not t.is_alive() for t in threads));self.assertEqual(len(result),world)
            return result

    def test_four_rank_real_admission(self):self.assertEqual(self.run_world(),['pass']*4)
    def test_four_rank_parent_deny(self):self.assertEqual(self.run_world(deny=True),['MemoryError']*4)
    def test_seventy_two_rank_step_mismatch_fails_every_rank(self):
        self.assertEqual(self.run_world(world=72,step_mismatch=True),['MemoryError']*72)
    def test_seventy_two_rank_initial_batch_participates_uniformly(self):
        self.assertEqual(self.run_world(world=72,initial=True),['pass']*72)
    def test_seventy_two_rank_local_error_fails_every_rank(self):
        self.assertEqual(self.run_world(world=72,local_failure=True),['MemoryError']*72)

if __name__=='__main__':unittest.main()
