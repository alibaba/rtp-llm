"""Fail-closed pre/post slot-reuse evidence schema tests."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from validate_world72_core_evidence import validate_first_reuse_evidence,Validator

def fixture():
    grad=dict(gradient_numel=6,gradient_count=3,all_gradients_clipped=True,retained_bytes=8,cpu_spilled_bytes=16,
              microbatch_contributions_min=2,microbatch_contributions_max=2)
    cpu=dict(name='cpu',tier='cpu',numel=4,bytes=16,device='cpu',storage_nbytes=16,storage_offset=0,
        storage_use_count=2,tensor_use_count=1,storage_id=101,storage_pointer=4096,contributions=2)
    gpu=dict(name='gpu',tier='gpu',numel=2,bytes=8,contributions=2)
    empty=dict(name='empty',tier='empty',numel=0,bytes=0,slot_present=True,contributions=2)
    plan=dict(generation=5,reusable_bytes=16,planned_moment_bytes=24,slots=[cpu,gpu,empty])
    pre=dict(schema='q38-first-adam-pre-update-gradient-v1',run_id='run',rank=4,pid=55,world_size=72,
        recorded_unix=100.,generation=5,expected_local_numel=6,gradient_store=grad,slot_plan=plan,
        captured_before_any_optimizer_write=True,tensor_payload_included=False)
    consumed=dict(name='cpu',gradient_bytes=16,storage_id=101,storage_pointer=4096,contributions=2,
        all_chunks_read_before_overwrite=True,first_dtype='bfloat16',second_dtype='bfloat16',first_stride=[2],second_stride=[2])
    commit=dict(complete=True,pre_update_evidence={'pre_update_sha256':'a'*64},pre_update_generation=5,
        consumed_gradients_still_resident_as_fp32=False,consumed_cpu_gradient_bytes=16,consumed_slots=[consumed])
    opt=dict(first_adam_reuse_commit=commit,gradient_store=dict(cpu_spilled_bytes=0,consumed_gradient_slots=1,gradient_numel=2),
             moments=dict(cpu_moment_bytes_reusing_gradient=16))
    return pre,opt

class ValidationTest(unittest.TestCase):
    def check(self,pre,opt):return validate_first_reuse_evidence(pre,'a'*64,opt,rank=4,pid=55,loaded_numel=6,update_time_ns=101_000_000_000,expected_run_id='run')
    def test_actual_pre_post_shape_passes(self):
        pre,opt=fixture();self.assertIs(self.check(pre,opt),pre['gradient_store'])
    def test_tampering_or_ambiguous_ownership_denied(self):
        cases=[lambda p,o:p.update(pid=56),lambda p,o:p.update(generation=6),lambda p,o:p.update(run_id='wrong-run'),
            lambda p,o:p['gradient_store'].update(gradient_numel=5),
            lambda p,o:p['slot_plan']['slots'][0].update(storage_use_count=3),
            lambda p,o:o['first_adam_reuse_commit']['consumed_slots'][0].update(storage_pointer=4098),
            lambda p,o:o['first_adam_reuse_commit']['consumed_slots'][0].update(all_chunks_read_before_overwrite=False),
            lambda p,o:o['gradient_store'].update(gradient_numel=6),
            lambda p,o:o['moments'].update(cpu_moment_bytes_reusing_gradient=15),
            lambda p,o:p['gradient_store'].update(microbatch_contributions_max=1)]
        for mutate in cases:
            p,o=fixture();mutate(p,o)
            with self.assertRaises((ValueError,TypeError,KeyError)):self.check(p,o)

    def test_mode_commit_and_world_vote_cannot_disagree(self):
        for enabled,commit,expected in [(False,{},'reuse configuration'),(True,None,'reuse configuration'),(True,{},'world update/audit vote'),(False,None,'reuse configuration'),('true',None,'reuse mode must be a boolean')]:
            with self.subTest(enabled=enabled,commit=commit):
                v=Validator([],first_adam_reuse_required=True)
                record=dict(rank=0,pid=55,world_size=72,time_ns=101_000_000_000,phase='train',
                    optimizer=dict(steps=[1],first_adam_reuse_enabled=enabled,first_adam_reuse_commit=commit,gradient_store={}))
                v.records=lambda pattern: [(Path('/not-read'),record)] if pattern=='optimizer-step-rank*.json' else []
                v.source_loads_and_plans=lambda:{}
                result=v.run()
                self.assertFalse(result['core_checks_passed'])
                self.assertTrue(any(expected in error for error in result['errors']),result['errors'])

    def test_full_run_keeps_reuse_pre_for_micro_boundary_method(self):
        import test_micro_memory_records_cpu as micro
        records,loads,updates=micro.fixture()
        class Replay:
            valid=True
            current_root='/tmp/runs/run'
            def validate(self,loads):pass
            def permits_first_update(self,*args):return True
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,value in records:(root/name).write_text(json.dumps(value))
            for step in (1,2):
                for rank,(_,row) in updates[step].items():
                    row=copy.deepcopy(row);row['optimizer']['steps']=[step]
                    if step==1 and rank==0:row['optimizer']['first_adam_reuse_commit']={}
                    (root/f'optimizer-step-rank{rank}-{step}.json').write_text(json.dumps(row))
            pre=dict(rank=0,gradient_store=dict(microbatch_contributions_min=2,microbatch_contributions_max=2))
            (root/'pre-adam-reuse-rank0.json').write_text(json.dumps(pre))
            validator=Validator([root]);validator.initial_replay=Replay();validator.source_loads_and_plans=lambda:loads
            result=validator.run()
            self.assertIn('micro_memory_boundaries',result['details'])
            self.assertFalse(any('micro_memory_boundaries:' in error for error in result['errors']),result['errors'])
            self.assertIn(0,validator.reuse_pre)

    def test_world_vote_must_belong_to_validated_current_run(self):
        class Replay:
            valid=True
            current_root='/tmp/runs/run'
            def validate(self,loads):pass
            def permits_first_update(self,*args):return True
        for marker_run,should_pass in [('run',True),('wrong-run',False)]:
            v=Validator([],first_adam_reuse_required=True);v.initial_replay=Replay()
            row=dict(rank=0,pid=55,world_size=72,time_ns=101,phase='train',optimizer=dict(
                steps=[1],first_adam_reuse_enabled=True,first_adam_reuse_commit={},gradient_store={}))
            marker=dict(schema='q38-reuse-world-update-vote-v1',run_id=marker_run,rank=0,pid=55,
                world_size=72,step=1,recorded_ns=102,post_update_votes=[dict(rank=i,phase='post-update-audit-and-clear',passed=True) for i in range(72)])
            v.records=lambda pattern: ([(Path('/marker'),marker)] if pattern=='reuse-world-commit-rank*.json' else
                [(Path('/step'),row)] if pattern=='optimizer-step-rank*.json' else [])
            v.source_loads_and_plans=lambda:{}
            result=v.run()
            errors=[e for e in result['errors'] if 'world update/audit vote' in e]
            self.assertEqual(bool(errors),not should_pass,result['errors'])

    def test_fresh_world_vote_binds_pinned_plan_without_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            plan_path=Path(folder)/'plan.json'
            plan=dict(run_id='run',run_root='/tmp/runs/run',environment=dict(
                Q38_RUN_ID='run',Q38_RUN_ROOT='/tmp/runs/run',Q38_FIRST_ADAM_REUSE_CPU_SPILL='1'))
            raw=json.dumps(plan).encode();plan_path.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
            for marker_run,pinned_hash,should_pass in [('run',sha,True),('other',sha,False),('run','0'*64,False)]:
                with self.subTest(marker_run=marker_run,pinned_hash=pinned_hash):
                    v=Validator([],first_adam_reuse_required=True,production_plan=plan_path,
                        production_plan_sha256=pinned_hash,expected_run_id='run')
                    row=dict(rank=0,pid=55,world_size=72,time_ns=101,phase='train',optimizer=dict(
                        steps=[1],first_adam_reuse_enabled=True,first_adam_reuse_commit={},gradient_store={}))
                    marker=dict(schema='q38-reuse-world-update-vote-v1',run_id=marker_run,rank=0,pid=55,
                        world_size=72,step=1,recorded_ns=102,post_update_votes=[dict(rank=i,
                        phase='post-update-audit-and-clear',passed=True) for i in range(72)])
                    v.records=lambda pattern: ([(Path('/marker'),marker)] if pattern=='reuse-world-commit-rank*.json' else
                        [(Path('/step'),row)] if pattern=='optimizer-step-rank*.json' else [])
                    v.source_loads_and_plans=lambda:{}
                    errors=v.run()['errors']
                    failures=[e for e in errors if 'world update/audit vote' in e]
                    self.assertEqual(bool(failures),not should_pass,errors)

    def test_replay_and_pinned_plan_must_identify_same_run(self):
        class Replay:
            valid=True
            current_root='/tmp/runs/run'
        v=Validator([]);v.initial_replay=Replay();v.validated_plan_run_id='run'
        self.assertEqual(v.reuse_run_identity(),'run')
        v.validated_plan_run_id='other'
        self.assertIsNone(v.reuse_run_identity())
        v.initial_replay.current_root='/tmp/runs/../other'
        self.assertIsNone(v.reuse_run_identity())

    def test_actual_reuse_mode_must_match_pinned_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            plan_path=Path(folder)/'plan.json'
            plan=dict(run_id='run',run_root='/tmp/runs/run',environment=dict(
                Q38_RUN_ID='run',Q38_RUN_ROOT='/tmp/runs/run',Q38_FIRST_ADAM_REUSE_CPU_SPILL='0'))
            raw=json.dumps(plan).encode();plan_path.write_bytes(raw)
            v=Validator([],production_plan=plan_path,production_plan_sha256=hashlib.sha256(raw).hexdigest(),
                expected_run_id='run')
            row=dict(rank=0,pid=55,world_size=72,time_ns=101,phase='train',optimizer=dict(
                steps=[1],first_adam_reuse_enabled=True,first_adam_reuse_commit={},gradient_store={}))
            v.records=lambda pattern:[(Path('/step'),row)] if pattern=='optimizer-step-rank*.json' else []
            v.source_loads_and_plans=lambda:{}
            self.assertTrue(any('actual reuse mode differs' in e for e in v.run()['errors']))

    def test_fresh_micro_boundaries_bind_all_current_loads_and_budget(self):
        import test_micro_memory_records_cpu as micro
        from validate_world72_core_evidence import GIB
        with tempfile.TemporaryDirectory() as folder:
            plan_path=Path(folder)/'plan.json'
            root='/tmp/runs/run';budget=72*GIB
            plan=dict(run_id='run',run_root=root,gradient_cache_bytes=budget,environment=dict(
                Q38_RUN_ID='run',Q38_RUN_ROOT=root,Q38_GPU_GRADIENT_CACHE_BYTES=str(budget),
                Q38_INITIAL_POLICY_REPLAY='0',Q38_REPLAY_VERIFY_FULL_SHARD_BYTES='1',
                Q38_FIRST_ADAM_REUSE_CPU_SPILL='1'))
            raw=json.dumps(plan).encode();plan_path.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
            def check(case):
                records,loads,updates=micro.fixture()
                source=dict(source_coordinates_verified=True,full_destination_BF16_bytes_verified=True,
                    local_numel=1,local_bytes=2,source_reads=1,full_slice_bytes_sha256=['a'*64])
                for rank in range(72):
                    loads[rank][1].update(schema='qwen38-flat-sharded-checkpoint-v1',rank=rank,world_size=72,
                        run_id='run',run_root=root,hostname='node'+str(rank//4),pid=1000+rank,
                        started_unix=1e-8,finished_unix=1e-7,source_provider='platform_mounted',
                        runtime_parameters=7324,source_slices=[source]*7324)
                for _,row in records:
                    if 'gradient_store' in row:row['gradient_store']['retained_budget_bytes']=budget
                if case=='missing-rank':del loads[71]
                elif case=='duplicate-identity':loads[1][1]['pid']=loads[0][1]['pid']
                elif case=='wrong-load-run':loads[0][1]['run_id']='old'
                elif case=='wrong-load-pid':loads[0][1]['pid']=9999
                elif case=='wrong-update-pid':updates[2][0][1]['pid']=9999
                elif case=='wrong-budget':micro.row(records,rank=0,actual_training_micro_index=1,event='enter-forward')['gradient_store']['retained_budget_bytes']=75*GIB
                elif case=='bad-source':loads[0][1]['source_slices']=[dict(source,full_destination_BF16_bytes_verified=False)]*7324
                elif case=='unread-source':loads[0][1]['source_slices']=[dict(source,source_reads=0,full_slice_bytes_sha256=[])]*7324
                elif case=='early-micro':loads[0][1]['finished_unix']=1.0
                elif case=='duplicate-rank':loads[1][1]['rank']=0
                v=Validator([],production_plan=plan_path,production_plan_sha256='0'*64 if case=='bad-hash' else sha,
                    expected_run_id='run')
                v.records=lambda pattern:[(Path(name),row) for name,row in records if Path(name).match(pattern)]
                v.micro_memory_boundaries(updates,loads)
                return v.errors,v.details
            errors,details=check('valid')
            self.assertEqual(errors,[])
            self.assertEqual([row['ranks_with_micro_records'] for row in details['micro_memory_boundaries']],[72,72])
            for case in ('missing-rank','duplicate-identity','wrong-load-run','wrong-load-pid','wrong-update-pid',
                    'wrong-budget','bad-source','unread-source','early-micro','duplicate-rank','bad-hash'):
                with self.subTest(case=case):self.assertTrue(check(case)[0])

    def test_pinned_plan_requires_matching_cli_and_valid_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'plan.json'
            def run(mode,required,*,wrong_hash=False,wrong_run=False):
                plan=dict(run_id='run',run_root='/tmp/runs/run',environment=dict(
                    Q38_RUN_ID='run',Q38_RUN_ROOT='/tmp/runs/run',Q38_FIRST_ADAM_REUSE_CPU_SPILL=mode))
                raw=json.dumps(plan).encode();path.write_bytes(raw)
                sha='0'*64 if wrong_hash else hashlib.sha256(raw).hexdigest()
                v=Validator([],first_adam_reuse_required=required,production_plan=path,
                    production_plan_sha256=sha,expected_run_id='other' if wrong_run else 'run')
                v.source_loads_and_plans=lambda:{}
                result=v.run()
                return v,[e for e in result['errors'] if 'reuse' in e.lower() and
                    ('plan' in e.lower() or 'cli flag' in e.lower())]
            for mode,required in [('1',True),('0',False)]:
                _,errors=run(mode,required);self.assertEqual(errors,[])
            v,errors=run('1',False);self.assertTrue(errors);self.assertTrue(v.first_adam_reuse_required)
            for mode,required in [('0',True),('yes',True),(1,True)]:
                _,errors=run(mode,required);self.assertTrue(errors,(mode,required))
            for kwargs in [dict(wrong_hash=True),dict(wrong_run=True)]:
                _,errors=run('1',True,**kwargs);self.assertTrue(errors)
            malformed=json.dumps(dict(run_id='run',run_root='/tmp/runs/run',environment='not-a-mapping')).encode()
            path.write_bytes(malformed)
            v=Validator([],first_adam_reuse_required=True,production_plan=path,
                production_plan_sha256=hashlib.sha256(malformed).hexdigest(),expected_run_id='run')
            v.source_loads_and_plans=lambda:{}
            self.assertTrue(any('environment must be a mapping' in e for e in v.run()['errors']))
            v=Validator([],first_adam_reuse_required=True);v.source_loads_and_plans=lambda:{}
            self.assertTrue(any('needs a pinned current-run plan' in e for e in v.run()['errors']))

if __name__=='__main__':unittest.main()
