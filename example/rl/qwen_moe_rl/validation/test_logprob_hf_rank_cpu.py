"""CPU-only tests of the actual all-rank probe method and source guard."""
import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from types import SimpleNamespace

root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'training/qwen38_policy_probe.py').read_text())
method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_policy_before_rollout')
class Dist:
    bad=None
    def get_rank(self):return 0
    def get_world_size(self):return 72
    def all_gather_object(self,result,row):
        for i in range(72):result[i]=dict(row,rank=i,raw_log_probs=list(row['raw_log_probs']))
        result[-1]['raw_log_probs']=self.bad
dist=Dist()
scope=dict(dist=dist,torch=SimpleNamespace(device=lambda *a:'stub',cuda=SimpleNamespace(current_device=lambda:0)),
    teacher_forced_target_scores=lambda *a:[-1.,-2.],os=os,math=math,Path=Path,struct=struct,json=json,time=time)
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'actual-policy-probe-AST','exec'),scope)
spec=importlib.util.spec_from_file_location('refit',root/'adapter/qwen38_refit_contract.py')
refit=importlib.util.module_from_spec(spec);spec.loader.exec_module(refit)
sync=dict(global_step=0,model_version='v',native_boot=0,native_incarnation='i',logical_model_replicas=1,
          received_weights=1590,ranks=list(range(72)),full_source_sha256='a'*64)
def engine():return SimpleNamespace(_q38_phases=SimpleNamespace(phase='train'),optimizer=SimpleNamespace(audit=lambda:{'steps':[]}),
    _q38_gradient_store=SimpleNamespace(slots=[]),module=SimpleNamespace(parameters=lambda:[]),
    _q38_hf_probe_tokenizer=SimpleNamespace(encode=lambda text,**kw: [1] if text.startswith('The value') else [2,3]))
with tempfile.TemporaryDirectory() as tmp:
    os.environ['Q38_TRAINING_AUDIT_DIR']=tmp
    os.environ.pop('Q38_LOGPROB_REPORT_ONLY',None);dist.bad=[-1.5,-2.]
    try:scope['audit_policy_before_rollout'](engine());raise AssertionError('strict must reject')
    except RuntimeError:pass
    os.environ['Q38_LOGPROB_REPORT_ONLY']='1';scope['audit_policy_before_rollout'](engine())
    hf=json.loads((Path(tmp)/'hf-probe-step0-rank0.json').read_text())
    assert hf['max_HF_rank_error']==.5 and hf['all_rank_HF_probe_agreement'] is False
    assert refit.validate_policy_probe_sources(sync,hf,step=0,version='v',boot=0,incarnation='i')==hf
    for key,value in [('max_HF_rank_error',float('nan')),('all_rank_HF_probe_agreement',True),('HF_rank_comparison_mode','strict')]:
        bad=copy.deepcopy(hf);bad[key]=value
        try:refit.validate_policy_probe_sources(sync,bad,step=0,version='v',boot=0,incarnation='i');raise AssertionError(key)
        except ValueError:pass
    for bad_scores in ([-1.], [float('nan'),-2.]):
        dist.bad=bad_scores
        try:scope['audit_policy_before_rollout'](engine());raise AssertionError('malformed scores accepted')
        except FloatingPointError:pass
print('WORLD72_HF_RANK_REPORT_ONLY_CPU_PASS')
