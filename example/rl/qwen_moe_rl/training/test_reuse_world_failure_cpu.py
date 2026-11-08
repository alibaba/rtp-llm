"""Exercise actual coordinated-update helper AST with 4 local mock ranks."""
import ast,os,math,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from test_distributed_admission import FakeDist

tree=ast.parse((Path(__file__).parent/'qwen38_bf16_training_runtime.py').read_text())
names={'_optimizer_step_after_global_clip','_poison_reuse_engine','_require_unpoisoned','_reuse_world_vote','_run_reuse_update'}
nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
scope=dict(os=os,math=math,WORLD_SIZE=4)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-reuse-update-AST','exec'),scope)

class FailureTest(unittest.TestCase):
    def exercise(self,fail_stage=None):
        dist=FakeDist(4);results=[];engines=[]
        for rank in range(4):
            def step(rank=rank):
                if fail_stage=='step' and rank==2:raise RuntimeError('partial update')
            def zero_grad(rank=rank):
                if fail_stage=='clear' and rank==2:raise RuntimeError('clear failed')
            opt=SimpleNamespace(step=step,zero_grad=zero_grad,changed_weight_elements=1,_reuse_failed=False)
            engines.append(SimpleNamespace(optimizer=opt,_q38_cpu_admission_group=object(),
                _q38_phases=SimpleNamespace(phase='train'),_q38_gradient_store=SimpleNamespace(_reuse_failed=False,_reuse_frozen=False)))
        def audit(engine,boundary):
            if fail_stage=='audit' and dist.get_rank()==2:raise OSError('audit failed')
        def run(rank):
            dist.local.rank=rank
            def reduce_changed(value):
                if fail_stage=='reduce' and rank==2:raise RuntimeError('reduce failed')
                return value
            try:
                def commit_writer(engine,rank,world,votes):
                    if fail_stage=='commit' and rank==2:raise OSError('commit receipt failed')
                    return 'record'
                scope['_run_reuse_update'](engines[rank],1.,reduce_changed,audit,dist,4,commit_writer=commit_writer)
                results.append((rank,'passed'))
            except BaseException as e:results.append((rank,type(e).__name__))
        with patch.dict(os.environ,{'RLLM_REQUIRE_REAL_UPDATE':'1'}):
            threads=[threading.Thread(target=run,args=(rank,)) for rank in range(4)]
            for t in threads:t.start()
            for t in threads:t.join(8)
        self.assertTrue(all(not t.is_alive() for t in threads));self.assertEqual(len(results),4)
        for engine in engines:
            if fail_stage:
                self.assertTrue(engine._q38_update_poisoned and engine.optimizer._reuse_failed and engine._q38_gradient_store._reuse_failed)
                self.assertEqual(engine._q38_phases.phase,'failed')
                with self.assertRaises(RuntimeError):scope['_require_unpoisoned'](engine)
            else:scope['_require_unpoisoned'](engine)
        return sorted(value for _,value in results)
    def test_all_successful(self):self.assertEqual(self.exercise(),['passed']*4)
    def test_partial_step_failure_world_poisoned(self):self.assertEqual(self.exercise('step'),['RuntimeError']*4)
    def test_one_audit_failure_world_poisoned(self):self.assertEqual(self.exercise('audit'),['RuntimeError']*4)
    def test_one_reduce_failure_world_poisoned(self):self.assertEqual(self.exercise('reduce'),['RuntimeError']*4)
    def test_one_gradient_clear_failure_world_poisoned(self):self.assertEqual(self.exercise('clear'),['RuntimeError']*4)
    def test_one_commit_receipt_failure_world_poisoned(self):self.assertEqual(self.exercise('commit'),['RuntimeError']*4)
    def test_first_step_stays_frozen_through_world_commit_vote(self):
        class SingleRank:
            def get_rank(self):return 0
            def all_gather_object(self,outputs,item,group=None):
                self.assert_frozen(item)
                outputs[0]=item
            def assert_frozen(self,item):
                assert store._reuse_frozen, item['phase']
        store=SimpleNamespace(_reuse_frozen=True,_reuse_failed=False)
        events=[]
        def clear():
            assert store._reuse_frozen
            events.append('cleared')
        def release():
            assert store._reuse_frozen
            events.append('released');store._reuse_frozen=False
        opt=SimpleNamespace(step=lambda:None,clear_first_reuse_before_world_vote=clear,
            release_first_reuse_after_world_commit=release,changed_weight_elements=1,_reuse_failed=False)
        engine=SimpleNamespace(optimizer=opt,_q38_cpu_admission_group=object(),
            _q38_phases=SimpleNamespace(phase='train'),_q38_gradient_store=store)
        def writer(*args):
            assert store._reuse_frozen and events==['cleared']
            return 'record'
        with patch.dict(os.environ,{'RLLM_REQUIRE_REAL_UPDATE':'1'}):
            scope['_run_reuse_update'](engine,1.,lambda x:x,lambda *args:None,SingleRank(),1,commit_writer=writer)
        self.assertEqual(events,['cleared','released']);self.assertFalse(store._reuse_frozen)
    def test_collective_timeout_poisons_without_claiming_rollback(self):
        class FailedDist:
            def get_rank(self):return 0
            def all_gather_object(self,*args,**kwargs):raise TimeoutError('unavailable peer')
        engine=SimpleNamespace(optimizer=SimpleNamespace(step=lambda:None,zero_grad=lambda:None,
            changed_weight_elements=1,_reuse_failed=False),_q38_cpu_admission_group=object(),
            _q38_phases=SimpleNamespace(phase='train'),_q38_gradient_store=SimpleNamespace(_reuse_failed=False,_reuse_frozen=False))
        with patch.dict(os.environ,{'RLLM_REQUIRE_REAL_UPDATE':'1'}):
            with self.assertRaises(TimeoutError):
                scope['_run_reuse_update'](engine,1.,lambda x:x,lambda *args:None,FailedDist(),4,commit_writer=lambda *args:'record')
        self.assertTrue(engine._q38_update_poisoned and engine.optimizer._reuse_failed)
        self.assertEqual(engine._q38_phases.phase,'failed')

if __name__=='__main__':unittest.main()
