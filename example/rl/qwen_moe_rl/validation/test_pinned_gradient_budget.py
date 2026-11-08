"""Offline metadata corruption tests; do not imply an RL/GPU qualification."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from validate_world72_core_evidence import GIB, Validator

class PinnedBudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.run='q38-test-current-run';self.run_root='/home/admin/test/runs/'+self.run
        self.plan={'run_id':self.run,'run_root':self.run_root,'gradient_cache_bytes':68*GIB,
                   'environment':{'Q38_RUN_ID':self.run,'Q38_RUN_ROOT':self.run_root,
                                  'Q38_GPU_GRADIENT_CACHE_BYTES':str(68*GIB)}}
        self.plan_path=self.root/'plan.json';self.write_plan()
        self.loads={rank:(None,{'loaded_local_numel':100}) for rank in range(72)}
        self.updates={step:{rank:(None,{'pid':1000+rank,'time_ns':10000,'optimizer':{'gradient_store':{
            'microbatch_contributions_min':2,'microbatch_contributions_max':2}}}) for rank in range(72)} for step in (1,2)}
        self.records=[]
        for step in (1,2):
            self.records.append((f'global-minibatch-step{step}.json',{'trainer_global_step':step,'rows_per_rank':2}))
            for rank in range(72):
                for local in range(2):
                    micro=(step-1)*2+local+1
                    store={'stored_gradient_dtype':'float32','gradient_numel':0 if local==0 else 100,
                           'microbatch_contributions_min':local,'microbatch_contributions_max':local,
                           'retained_budget_bytes':68*GIB,'retained_bytes':0 if local==0 else 400,'cpu_spilled_bytes':0}
                    for event in ('before-cache-release','enter-forward','exit-forward-before-backward'):
                        self.records.append((f'micro-memory-rank{rank}-{event}-{micro}.json',{
                            'schema':'q38-actual-engine-micro-memory-boundary-v1','world_size':72,'rank':rank,
                            'optimizer_steps':[] if step==1 else [1],'actual_training_micro_index':micro,
                            'event':event,'pid':1000+rank,'time_ns':1000+micro,'parameter_shadow_grad_count':0,
                            'parameter_shadow_grad_bytes':0,'gradient_store':copy.deepcopy(store),
                            'tensors_loss_rewards_gradients_modified':False,'CPU_cgroup_limit_or_peak_inferred':False,
                            'empty_allocator_cache_released':event=='enter-forward'}))
        for name,row in self.records:(self.root/name).write_text(json.dumps(row))
        (self.root/'initial-policy-replay-world72.json').write_text(json.dumps({'current_run_root':self.run_root}))
        for rank in range(72):
            (self.root/f'initial-policy-replay-rank{rank}.json').write_text(json.dumps({
                'schema':'q38-fresh-initial-policy-replay-worker-v1','rank':rank,'world_size':72,
                'current_run_root':self.run_root,'pid':1000+rank}))

    def tearDown(self):self.tmp.cleanup()
    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan));self.sha=hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
    def check(self,*,pinned=True,sha=None,run=None,valid_replay=True):
        args={} if not pinned else {'production_plan':self.plan_path,'production_plan_sha256':sha or self.sha,
                                    'expected_run_id':run or self.run}
        validator=Validator([self.root],**args);validator.initial_replay=SimpleNamespace(valid=valid_replay)
        validator.micro_memory_boundaries(self.updates,self.loads)
        return validator
    def rewrite(self,fn):
        for name,_ in self.records:
            if name.startswith('micro-memory-'):
                path=self.root/name;row=json.loads(path.read_bytes());fn(row);path.write_text(json.dumps(row))
    def test_exact_pinned_68_passes(self):
        v=self.check();self.assertEqual(v.errors,[]);self.assertEqual(v.details['gradient_cache_budget_source']['bytes'],68*GIB)
    def test_historical_75_default(self):
        self.rewrite(lambda r:r['gradient_store'].update(retained_budget_bytes=75*GIB))
        v=self.check(pinned=False);self.assertEqual(v.errors,[]);self.assertEqual(v.details['gradient_cache_budget_source']['kind'],'historical_default')
    def test_68_without_plan_is_not_default(self):self.assertTrue(self.check(pinned=False).errors)
    def test_wrong_evidence_budget_rejected(self):
        self.rewrite(lambda r:r['gradient_store'].update(retained_budget_bytes=75*GIB) if r['rank']==0 else None)
        self.assertTrue(self.check().errors)
    def test_forged_plan_bytes_rejected_by_trusted_hash(self):
        original=self.sha;self.plan['gradient_cache_bytes']=75*GIB;self.write_plan()
        self.assertTrue(self.check(sha=original).errors)
    def test_self_inconsistent_plan_rejected_even_with_matching_hash(self):
        self.plan['gradient_cache_bytes']=75*GIB;self.write_plan();self.assertTrue(self.check().errors)
    def test_wrong_run_id_rejected(self):self.assertTrue(self.check(run='unrelated-run').errors)
    def test_other_world_root_rejected(self):
        (self.root/'initial-policy-replay-world72.json').write_text(json.dumps({'current_run_root':'/other/runs/'+self.run}))
        self.assertTrue(self.check().errors)
    def test_missing_rank_replay_rejected(self):
        (self.root/'initial-policy-replay-rank71.json').unlink();self.assertTrue(self.check().errors)
    def test_other_run_rank_replay_rejected(self):
        p=self.root/'initial-policy-replay-rank71.json';d=json.loads(p.read_bytes());d['current_run_root']='/other/runs/'+self.run;p.write_text(json.dumps(d));self.assertTrue(self.check().errors)
    def test_wrong_pid_rejected(self):
        p=self.root/'initial-policy-replay-rank71.json';d=json.loads(p.read_bytes());d['pid']=999;p.write_text(json.dumps(d));self.assertTrue(self.check().errors)
    def test_unvalidated_replay_rejected(self):self.assertTrue(self.check(valid_replay=False).errors)
    def test_missing_plan_component_rejected(self):
        v=Validator([self.root],production_plan=self.plan_path,expected_run_id=self.run);v.initial_replay=SimpleNamespace(valid=True)
        v.micro_memory_boundaries(self.updates,self.loads);self.assertTrue(v.errors)

if __name__=='__main__':unittest.main()
