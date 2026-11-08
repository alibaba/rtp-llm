"""Best-effort score reporting; never creates training acceptance errors."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct


def _scores(value):
    if not isinstance(value,list) or not value:return None
    try:
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in value):return None
        # Scores are claimed as FP32; unrepresentable values are not reported
        # as if they were valid FP32 scores.
        struct.pack('<%df'%len(value),*value)
    except (OverflowError,ValueError,struct.error):return None
    return value


def _read(roots,pattern,issues):
    for root in roots:
        try:paths=sorted(Path(root).rglob(pattern))
        except OSError as error:
            issues.append({'pattern':pattern,'error':type(error).__name__});continue
        for path in paths:
            try:
                if not path.is_file() or path.stat().st_size>64*1024**2:
                    issues.append({'path':str(path),'error':'not_regular_or_too_large'});continue
                raw=path.read_bytes();row=json.loads(raw)
                if not isinstance(row,dict):raise ValueError('score JSON is not an object')
                yield str(path),hashlib.sha256(raw).hexdigest(),row
            except (OSError,ValueError,TypeError,UnicodeError) as error:
                issues.append({'path':str(path),'error':type(error).__name__})


def _tokens(value):
    return value if isinstance(value,list) and all(type(x) is int for x in value) else None


def _bits_match(values,bits):
    if not isinstance(bits,str):return None
    try:return struct.pack('<%df'%len(values),*values).hex()==bits.lower()
    except (OverflowError,ValueError,struct.error):return False


def report(roots,syncs,*,historical_roots=(),historical_policy0=False):
    """Record available values and comparisons without altering validation."""
    issues=[];hf={};native={}
    def source_roots(step):
        return historical_roots if historical_policy0 and step==0 else roots
    for step in (0,1,2):
        for path,sha,row in _read(source_roots(step),f'hf-probe-step{step}-rank*.json',issues):
            rank=row.get('rank');values=_scores(row.get('raw_log_probs'))
            if type(rank) is not int or not 0<=rank<72 or row.get('global_step')!=step or values is None:
                issues.append({'path':path,'error':'invalid_hf_score_record'});continue
            bits=row.get('raw_fp32_bits_hex')
            hf.setdefault(step,[]).append({'path':path,'sha256':sha,'rank':rank,'scores':values,
                'raw_fp32_bits_hex':bits if isinstance(bits,str) else None,
                'reported_bits_match_values':_bits_match(values,bits),
                'prompt_ids':_tokens(row.get('prompt_ids')),'response_ids':_tokens(row.get('response_ids'))})
        native_records=itertools.chain(
            _read(source_roots(step),f'rtp_probe_rank0_step{step}.json',issues),
            _read(source_roots(step),f'rtp_probe_partial_step{step}_*.json',issues))
        for path,sha,row in native_records:
            if row.get('global_step')!=step:
                issues.append({'path':path,'error':'native_step_mismatch'});continue
            routes=row.get('probes')
            if not isinstance(routes,list):issues.append({'path':path,'error':'invalid_native_routes'});continue
            for route in routes:
                if not isinstance(route,dict):issues.append({'path':path,'error':'invalid_native_route'});continue
                dp=route.get('dp_rank');values=_scores(route.get('raw_log_probs'))
                if type(dp) is not int or not 0<=dp<9 or values is None:
                    issues.append({'path':path,'error':'invalid_native_score'});continue
                bits=route.get('raw_fp32_bits_hex')
                native.setdefault(step,[]).append({'path':path,'sha256':sha,'dp_rank':dp,'scores':values,
                    'raw_fp32_bits_hex':bits if isinstance(bits,str) else None,
                    'reported_bits_match_values':_bits_match(values,bits),
                    'model_version':row.get('model_version') if isinstance(row.get('model_version'),str) else None,
                    'full_source_sha256':row.get('full_source_sha256') if isinstance(row.get('full_source_sha256'),str) else None,
                    'prompt_ids':_tokens(row.get('prompt_ids')),'response_ids':_tokens(row.get('response_ids'))})
    summary=[]
    for step in (0,1,2):
        references=hf.get(step,[]);routes=native.get(step,[]);sync=syncs.get(step,{})
        comparisons=[]
        for reference in references:
            for route in routes:
                a,b=reference['scores'],route['scores']
                if (len(a)!=len(b) or reference['prompt_ids'] is None or reference['response_ids'] is None
                        or len(reference['response_ids'])!=len(a) or reference['prompt_ids']!=route['prompt_ids']
                        or reference['response_ids']!=route['response_ids']
                        or route['model_version']!=sync.get('model_version')
                        or route['full_source_sha256']!=sync.get('full_source_sha256')):continue
                differences=[abs(x-y) for x,y in zip(a,b)]
                if all(math.isfinite(x) for x in differences):
                    comparisons.append({'hf_rank':reference['rank'],'dp_rank':route['dp_rank'],
                        'max_abs_error':max(differences),'mean_abs_error':sum(differences)/len(differences)})
        summary.append({'step':step,'hf_scores':references,'native_scores':routes,'comparisons':comparisons,
                        'used_for_acceptance':False})
    finals=[]
    for path,sha,row in _read(roots,'rtp_final_policy2_actual_reprobe.json',issues):
        finals.append({'path':path,'sha256':sha,'global_step':row.get('global_step') if type(row.get('global_step')) is int else None,
                       'model_version':row.get('model_version') if isinstance(row.get('model_version'),str) else None})
    return {'steps':summary,'final_reprobe_records':finals,'issues':issues,'used_for_acceptance':False}
