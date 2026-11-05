"""CPU checks for the delivered action budget; never a GPU/RL acceptance."""
from pathlib import Path
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validator',type=Path,default=Path(__file__).with_name('validate_rl_run.py'))
    parser.add_argument('--counterexample',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    tree=ast.parse(args.validator.read_text())
    nodes=[n for n in tree.body if
        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('ACTION_') for t in n.targets)
        or isinstance(n,ast.FunctionDef) and n.name=='episode_action_budget']
    scope={'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(args.validator),'exec'),scope)
    check=scope['episode_action_budget']
    def fixture(lengths):
        prompts=[520,2050,2664,1][:len(lengths)]
        return dict(episode_id='synthetic-cpu-fixture-not-an-RL-episode',
            trajectories=[dict(steps=[dict(prompt_ids=[2]*p,response_ids=[1]*n) for p,n in zip(prompts,lengths)])],
            metrics=dict(agent_completion_tokens=sum(lengths),steps_used=len(lengths),steps_collected=len(lengths),
                num_trajectories=1,mean_response_len=sum(lengths)/len(lengths),max_response_len=max(lengths),
                min_response_len=min(lengths),max_prompt_len=max(prompts),min_prompt_len=min(prompts)))
    cases=[]
    for name,lengths,expected in [('boundary3072',[1374,436,1262],True),
            ('old3858',[1374,436,2048],False),('call2049',[2049],False),
            ('fourturns',[768,768,768,768],False)]:
        result=check(fixture(lengths));assert result['passed'] is expected,(name,result)
        cases.append(dict(name=name,expected_passed=expected,observed=result))
    underreported=fixture([1374,436,1262]);underreported['metrics']['agent_completion_tokens']=3071
    result=check(underreported);assert not result['passed']
    cases.append(dict(name='underreported_usage',expected_passed=False,observed=result))
    method=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Validator'
                for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='delivered_trace_linkage')
    assert 'super().delivered_trace_linkage(c)' in ast.unparse(method)
    original=None
    if args.counterexample:
        raw=args.counterexample.read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='a8b78fb2ef8147247f9ebe44fab67194efb6385d9c23d9f96d2cbad5ae5e2fe8'
        episode=json.loads(raw);before=deepcopy(episode);result=check(episode)
        assert episode==before and not result['passed'] and result['actual_action_tokens']==3858
        assert [r['actual_action_tokens'] for r in result['calls']]==[1374,436,2048]
        assert result['calls'][2]['remaining_before']==1262
        original=dict(path=str(args.counterexample),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),result=result)
    proof=dict(schema='q397-delivered-action-budget-cpu-v1',passed=True,
        validator_sha256=hashlib.sha256(args.validator.read_bytes()).hexdigest(),
        synthetic_cases=cases,original_trace_super_check_retained=True,
        original_counterexample=original,source_data_rewritten=False,counts_as_rl_acceptance=False)
    if args.output:
        with args.output.open('x') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
    print('ACTION_BUDGET_CPU_PASS',json.dumps(dict(passed=True,synthetic_cases=len(cases),
        actual_counterexample_verified=original is not None,counts_as_rl_acceptance=False)),flush=True)


if __name__=='__main__':main()
