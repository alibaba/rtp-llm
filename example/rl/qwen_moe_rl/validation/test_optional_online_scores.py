"""CPU fixtures for optional diagnostics; no model, CUDA or real Ray."""
import ast
import asyncio
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PRIVATE=(HERE/'qwen38_world_server.py').exists()
REPO=Path(os.environ.get('Q38_TEST_REPO', '/Users/lilele/Desktop/Codex File/rtp-llm')) if PRIVATE else HERE.parents[3]
VALIDATION=REPO/'example/rl/qwen_moe_rl/validation'
ADAPTER=REPO/'example/rl/qwen_moe_rl/adapter'
SOURCE_ADAPTER=HERE if PRIVATE else ADAPTER
SOURCE_TRAINING=HERE if PRIVATE else REPO/'example/rl/qwen_moe_rl/training'
SOURCE_TOKEN=HERE/'token_rpc_client.py' if PRIVATE else REPO/'rtp_llm/RL/token_rpc_client.py'
sys.path.insert(0,str(ADAPTER));sys.path.insert(0,str(VALIDATION));sys.path.insert(0,str(HERE))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


class FakeDist:
    def __init__(self,world):
        self.world=world;self.local=threading.local();self.barrier=threading.Barrier(world,timeout=8)
        self.lock=threading.Lock();self.round=0;self.rows={}
    def get_world_size(self):return self.world
    def get_rank(self):return self.local.rank
    def all_gather_object(self,out,value):
        with self.lock:self.rows.setdefault(self.round,{})[self.local.rank]=value
        self.barrier.wait()
        with self.lock:out[:]=[self.rows[self.round][i] for i in range(self.world)]
        token=self.barrier.wait()
        if token==0:
            with self.lock:self.round+=1
        self.barrier.wait()


class HFModeTests(unittest.TestCase):
    def setUp(self):
        self.dist=FakeDist(4);torch=ModuleType('torch');torch.__path__=[]
        torch.no_grad=lambda:(lambda f:f);torch.device=lambda *x:x;torch.cuda=SimpleNamespace(current_device=lambda:0)
        distmod=ModuleType('torch.distributed')
        for name in ('get_world_size','get_rank','all_gather_object'):setattr(distmod,name,getattr(self.dist,name))
        torch.distributed=distmod
        self.modules=patch.dict(sys.modules,{'torch':torch,'torch.distributed':distmod});self.modules.start()
        self.mod=load('candidate_policy_probe',SOURCE_TRAINING/'qwen38_policy_probe.py')
    def tearDown(self):self.modules.stop()

    def exercise(self,kind,world=4):
        if world != 4:self.dist.world=world;self.dist.barrier=threading.Barrier(world,timeout=20)
        results=[];calls=[];engines=[]
        class Tokenizer:
            def __init__(self,rank):self.rank=rank
            def encode(self,*args,**kwargs):
                if kind=='missing' and self.rank==world-1:raise ValueError('missing optional tokenizer')
                return [1,2]
        for rank in range(world):
            engine=SimpleNamespace(_q38_phases=SimpleNamespace(phase='train'),
                optimizer=SimpleNamespace(audit=lambda:{'steps':[]}),
                _q38_gradient_store=SimpleNamespace(slots={}),module=SimpleNamespace(parameters=lambda:[]),
                _q38_hf_probe_tokenizer=Tokenizer(rank))
            if kind=='cached_mismatch' and rank==2:engine._q38_hf_probe_step=0
            engines.append(engine)
        def teacher(*args,**kwargs):
            rank=self.dist.get_rank();calls.append(rank)
            if kind=='none' and rank==2:return None
            return [float('nan'),-2.] if kind=='nan' and rank==2 else [-1.,-2.]
        self.mod.teacher_forced_target_scores=teacher
        def worker(rank):
            self.dist.local.rank=rank
            try:self.mod.audit_policy_before_rollout(engines[rank]);results.append((rank,'ok'))
            except BaseException as e:results.append((rank,type(e).__name__))
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,
                {'Q38_OPTIONAL_NUMERICAL_PROBES':'2','Q38_TRAINING_AUDIT_DIR':directory}):
            threads=[threading.Thread(target=worker,args=(rank,)) for rank in range(world)]
            for t in threads:t.start()
            for t in threads:t.join(12)
            self.assertTrue(all(not t.is_alive() for t in threads))
            files=list(Path(directory).glob('hf-probe-*.json'))
            records=[json.loads(p.read_text()) for p in files]
        return sorted(results),calls,records

    def test_four_rank_nan_is_report_unavailable_and_not_a_gate(self):
        rows,calls,records=self.exercise('nan')
        self.assertEqual([x[1] for x in rows],['ok']*4);self.assertEqual(len(calls),4)
        bad=next(r for r in records if r['rank']==2)
        self.assertFalse(bad['score_available']);self.assertIsNone(bad['raw_log_probs'])
        self.assertTrue(all(not r['all_rank_scores_available'] for r in records))

    def test_missing_logits_score_is_gathered_as_unavailable(self):
        rows,calls,records=self.exercise('none')
        self.assertEqual([x[1] for x in rows],['ok']*4);self.assertEqual(len(calls),4)
        self.assertTrue(all(not r['all_rank_scores_available'] for r in records))
        self.assertIsNone(next(r for r in records if r['rank']==2)['raw_log_probs'])

    def test_forward_errors_are_fatal_but_missing_output_is_optional(self):
        class TorchStub:
            int64='int64'
            tensor=staticmethod(lambda *args,**kwargs:object())
            ones_like=staticmethod(lambda *args:object())
        class Model:
            training=True
            def eval(self):self.training=False
            def train(self,previous):self.training=previous
            def __call__(self,**kwargs):return SimpleNamespace()
        original=load('candidate_actual_policy_probe',SOURCE_TRAINING/'qwen38_policy_probe.py')
        original.torch=TorchStub();model=Model()
        self.assertIsNone(original.teacher_forced_target_scores(model,[1],[2],'cuda',allow_invalid=True))
        self.assertTrue(model.training)
        with self.assertRaises(AttributeError):original.teacher_forced_target_scores(model,[1],[2],'cuda')
        class OutOfMemory(Model):
            def __call__(self,**kwargs):raise RuntimeError('CUDA out of memory')
        with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):
            original.teacher_forced_target_scores(OutOfMemory(),[1],[2],'cuda',allow_invalid=True)
        class InvalidForward(Model):
            def __call__(self,**kwargs):raise ValueError('actual forward failed')
        with self.assertRaisesRegex(ValueError,'actual forward failed'):
            original.teacher_forced_target_scores(InvalidForward(),[1],[2],'cuda',allow_invalid=True)

    def test_four_rank_missing_tokenizer_skips_forward_on_every_rank(self):
        rows,calls,records=self.exercise('missing')
        self.assertEqual([x[1] for x in rows],['ok']*4);self.assertEqual(calls,[]);self.assertEqual(records,[])

    def test_seventy_two_rank_missing_tokenizer_skips_forward_on_every_rank(self):
        rows,calls,records=self.exercise('missing',world=72)
        self.assertEqual([x[1] for x in rows],['ok']*72);self.assertEqual(calls,[]);self.assertEqual(records,[])

    def test_mode_cache_mismatch_fails_all_before_forward(self):
        rows,calls,_=self.exercise('cached_mismatch')
        self.assertEqual([x[1] for x in rows],['RuntimeError']*4);self.assertEqual(calls,[])

    def test_real_forward_or_collective_error_is_not_swallowed(self):
        self.mod.dist=SimpleNamespace(get_world_size=lambda:1,get_rank=lambda:0,
            all_gather_object=lambda output,value:output.__setitem__(0,value))
        engine=SimpleNamespace(_q38_phases=SimpleNamespace(phase='train'),optimizer=SimpleNamespace(audit=lambda:{'steps':[]}),
            _q38_gradient_store=SimpleNamespace(slots={}),module=SimpleNamespace(parameters=lambda:[]),
            _q38_hf_probe_tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2]))
        def fail(*a,**k):raise RuntimeError('simulated CUDA OOM')
        self.mod.teacher_forced_target_scores=fail
        with patch.dict(os.environ,{'Q38_OPTIONAL_NUMERICAL_PROBES':'2'}),self.assertRaisesRegex(RuntimeError,'CUDA OOM'):
            self.mod.audit_policy_before_rollout(engine)


def token_class():
    source=SOURCE_TOKEN.read_text();tree=ast.parse(source)
    wanted=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
            and n.name in ('TokenProtocolError','OptionalScoreDataError','_tensor_values')]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TokenRpcClient')
    method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='score_tokens')
    shell=ast.ClassDef(name='ScoreFixture',bases=[],keywords=[],decorator_list=[],body=[method])
    scope=dict(math=math,struct=struct,Optional=Optional,pb=SimpleNamespace(TensorPB=SimpleNamespace(FP32=1,INT32=2)),
               TokenSamplingParams=lambda **kwargs:SimpleNamespace(**kwargs))
    exec(compile(ast.fix_missing_locations(ast.Module(body=wanted+[shell],type_ignores=[])),'candidate-token-client','exec'),scope)
    return scope


class TokenErrorsTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_score_content_error_is_optional_subclass(self):
        scope=token_class();client=scope['ScoreFixture']();client._check_tokens=lambda x:None
        client._config=lambda x:SimpleNamespace(return_prompt_logits=False,return_target_logprob=False,
            prompt_logits_top_k=0,prompt_logits_start=0,prompt_logits_end=0)
        tensor=SimpleNamespace(data_type=1,shape=[1],fp32_data=struct.pack('<f',float('nan')))
        output=SimpleNamespace(start_pos=0,end_pos=1,target_logprobs=tensor)
        final=SimpleNamespace(HasField=lambda x:True,prompt_logits=output)
        response=SimpleNamespace(model_version='v',flatten_output=final)
        async def reply(*args):return response
        client._request=reply
        with self.assertRaises(scope['OptionalScoreDataError']):await client.score_tokens(1,[1],[2],expected_version='v')
        response.model_version='other'
        with self.assertRaises(scope['TokenProtocolError']) as e:await client.score_tokens(2,[1],[2],expected_version='v')
        self.assertNotIsInstance(e.exception,scope['OptionalScoreDataError'])
        async def network(*args):raise TimeoutError('actual transport failure')
        client._request=network
        with self.assertRaises(TimeoutError):await client.score_tokens(3,[1],[2],expected_version='v')

    def test_training_generate_implementation_is_unchanged(self):
        source=SOURCE_TOKEN.read_text();tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TokenRpcClient')
        method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='generate')
        self.assertEqual(hashlib.sha256(ast.get_source_segment(source,method).encode()).hexdigest(),
                         'f743da68d314124941f74ed1e10142c4f60bd19526bdc7f73d1b0f1b7d47e67c')


class OptionalScoreDataError(RuntimeError):pass


class WorldControllerTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,mode='2',issue=None,final=False):
        import test_qwen38_world_cpu as fixture
        previous=fixture.HERE;fixture.HERE=SOURCE_ADAPTER
        candidate_refit=load('qwen38_refit_contract',SOURCE_ADAPTER/'qwen38_refit_contract.py')
        token=ModuleType('rtp_llm.RL.token_rpc_client');token.OptionalScoreDataError=OptionalScoreDataError
        try:
            with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'Q38_OPTIONAL_NUMERICAL_PROBES':mode}),\
                    patch.dict(sys.modules,{'qwen38_refit_contract':candidate_refit,'rtp_llm.RL.token_rpc_client':token}),\
                    fixture.module(Path(folder)) as (mod,native):
                root=Path(folder);events=[];obj,_=fixture.controller(mod,root,events)
                step=2 if final else 0;version='inc:3' if final else 'inc:1'
                obj._started=obj._owners_active=True;obj.boot=0;obj.incarnation='inc'
                obj._committed_step=step;obj._committed_version=version
                obj.clients=[fixture.Client() for _ in range(9)]
                obj.tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2])
                fixture.fixed_probe_sources(root,step=step,version=version)
                sync_path=root/f'rtp_sync_rank0_step{step}.json';sync=json.loads(sync_path.read_text())
                sync['tensors']=[{'name':name,'sha256':'a'*64} for name in fixture.NAMES]
                if issue=='bad-sync':sync['tensors'].pop()
                sync_path.write_text(json.dumps(sync))
                # A missing or malformed HF diagnostic file is optional.
                if issue in ('missing-hf','bad-score','network','oom','bad-status','bad-sync','mode0'):
                    (root/'training'/f'hf-probe-step{step}-rank0.json').unlink()
                if issue=='bad-score':
                    async def invalid(*args,**kwargs):raise OptionalScoreDataError('completed response has invalid score')
                    obj.clients[2].score_tokens=invalid
                if issue in ('network','oom'):
                    async def fail(*args,**kwargs):raise TimeoutError('transport failure') if issue=='network' else RuntimeError('CUDA out of memory')
                    obj.clients[2].score_tokens=fail
                async def control(_topo,**kwargs):
                    value=fixture.Control(_topo,events,step=step,version=version)
                    if issue=='bad-status':value.replies.pop()
                    return value
                mod._control=control
                if issue in ('network','oom','bad-status','bad-sync'):
                    with self.assertRaises((RuntimeError,TimeoutError,ValueError)):
                        await obj.wake_and_probe(step,version)
                    self.assertFalse(obj._generation_ready);self.assertTrue(obj._incarnation_failed)
                    self.assertTrue(all(not c.open for c in obj.clients))
                    return None
                proof=await (obj.verify_final_policy(2) if final else obj.wake_and_probe(step,version))
                self.assertTrue(obj._generation_ready);self.assertTrue(all(c.open for c in obj.clients))
                if mode=='0':self.assertTrue(all(not c.entered.is_set() for c in obj.clients))
                else:self.assertTrue(all(c.entered.is_set() for c in obj.clients if c is not obj.clients[2] or issue!='bad-score'))
                if final:
                    finalmod=load('candidate_final_policy',SOURCE_TRAINING/'qwen38_final_policy.py')
                    self.assertIs(finalmod.validate_final_policy_proof(proof),proof)
                    bad=dict(proof,statuses=proof['statuses'][:-1])
                    with self.assertRaises(RuntimeError):finalmod.validate_final_policy_proof(bad)
                    bad=dict(proof,model_version='')
                    with self.assertRaises(RuntimeError):finalmod.validate_final_policy_proof(bad)
                return proof
        finally:fixture.HERE=previous

    async def test_missing_hf_and_invalid_native_score_are_non_gating(self):
        proof=await self.exercise(issue='missing-hf')
        self.assertIn('HF_score_unavailable',proof['diagnostic_issues']);self.assertEqual(len(proof['probes']),9)
        proof=await self.exercise(issue='bad-score',final=True)
        self.assertIn('native_DP2:OptionalScoreDataError',proof['diagnostic_issues']);self.assertEqual(len(proof['probes']),8)

    async def test_explicit_disable_does_not_issue_score_rpcs(self):
        proof=await self.exercise(mode='0',issue='mode0',final=True)
        self.assertTrue(proof['numerical_probe_skipped']);self.assertFalse(proof['final_native_GPU_reprobe'])

    async def test_transport_oom_and_refit_failures_stay_fatal(self):
        for issue in ('network','oom','bad-status','bad-sync'):
            with self.subTest(issue=issue):await self.exercise(issue=issue)


if __name__=='__main__':unittest.main()
