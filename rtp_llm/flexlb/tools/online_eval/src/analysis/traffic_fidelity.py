"""Standalone input-fidelity diagnostics using the production request generator.

Metrics describe infinite-history input structure, not finite-cache performance.
No alternate synthetic generator lives in this analysis module.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from traffic.datasets import distribution, trace_models
from traffic.prefix_lineage import BLOCK, decode
from traffic.structure import capture_shape, generated_shape
from traffic.calibrate_traffic import calibrate
from traffic.realistic import resolve, iter_requests

DEFAULT_THRESHOLDS = dict(length_ks=[.03, .10], joint_tv=[.15, .35], depth_ks=[.05, .25])
BINS = dict(length_log2_min=9, length_log2_max=21, length_bins=32,
            depth_max_blocks=2000, depth_bins=16, overflow='clamped_to_edge_bin')


def ks(a, b):
    if not a or not b:
        return None
    ca, cb = Counter(a), Counter(b)
    xa = xb = 0
    result = 0
    for value in sorted(ca.keys() | cb.keys()):
        xa += ca[value] / len(a)
        xb += cb[value] / len(b)
        result = max(result, abs(xa-xb))
    return result


def correlation(x, y):
    mx, my = sum(x)/len(x), sum(y)/len(y)
    vx = sum((v-mx)**2 for v in x)
    vy = sum((v-my)**2 for v in y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(vx*vy) if vx and vy else None


def histogram(shape):
    grid = [[0]*16 for _ in range(32)]
    for b,s in zip(shape['blocks'], shape['shared']):
        x = min(31,max(0,int((math.log2(b*BLOCK)-9)/12*32)))
        y = min(15,max(0,int(s/2000*16)))
        grid[x][y] += 1
    return grid


def summarize(shape):
    blocks, shared = shape['blocks'], shape['shared']
    n = len(blocks)
    return dict(requests=n, input_tokens=distribution([b*BLOCK for b in blocks]),
                depth_blocks=distribution(shared), cold_fraction=sum(s==0 for s in shared)/n,
                token_weighted_sharing=sum(shared)/sum(blocks),
                families=len(shape['families']), top5=sum(shape['families'][:5])/n,
                corr_log_length_depth=correlation([math.log2(b*BLOCK) for b in blocks],shared),
                length_counts=sorted(Counter(b*BLOCK for b in blocks).items()),
                depth_counts=sorted(Counter(shared).items()), joint_counts=histogram(shape))


def compare(real, generated):
    h1, h2 = histogram(real), histogram(generated)
    n1,n2=len(real['blocks']),len(generated['blocks'])
    return dict(length_ks=ks(real['blocks'],generated['blocks']),
                depth_ks=ks(real['shared'],generated['shared']),
                joint_tv=sum(abs(a/n1-b/n2) for r1,r2 in zip(h1,h2) for a,b in zip(r1,r2))/2)


def thresholds(value=None):
    result = dict(DEFAULT_THRESHOLDS, **(value or {}))
    if set(result) != set(DEFAULT_THRESHOLDS) or any(
        not isinstance(v, list) or len(v)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v)
        or not 0 <= v[0] <= v[1] <= 1 for v in result.values()):
        raise ValueError('thresholds require [pass_max, warn_max] in [0,1] for length_ks/joint_tv/depth_ks')
    return result


def grade(metrics, limits=None, valid=True):
    if not valid:
        return dict(length='UNASSESSED', cache='UNASSESSED', structure='UNASSESSED')
    limits = thresholds(limits)
    scores = {key: 0 if value <= limits[key][0] else 1 if value <= limits[key][1] else 2
              for key,value in metrics.items()}
    names=['PASS','WARN','FAIL']
    return dict(length=names[scores['length_ks']],
                structure=names[max(scores.values())],
                cache=names[max(1,max(scores.values()))])


def load_capture(path):
    path=Path(path); raw=path.read_bytes()
    metadata,events=decode(raw)
    return dict(path=str(path.resolve()),name=path.stem,sha256=hashlib.sha256(raw).hexdigest(),
                metadata=metadata,events=events,shape=capture_shape(events))


def audit(profile, capture):
    if capture is None:
        return dict(status='UNAVAILABLE',checks={},note='No capture matches calibration.model_sha256')
    calibration=profile.get('calibration',{})
    expected=calibrate_capture(capture)
    checks={}
    for key,value in expected['parameters'].items():
        if key=='joint_distribution' and profile.get('schema_version',1)==1:
            continue
        checks['parameters.'+key]=profile['parameters'].get(key)==value
    for key,value in expected['calibration']['targets'].items():
        checks['targets.'+key]=calibration.get('targets',{}).get(key)==value
    checks['source_sha256']=calibration.get('model_sha256')==capture['sha256']
    checks['provenance']=calibration.get('provenance')==capture['metadata']['provenance']
    embedded=capture['metadata']['provenance'].get('fit_report_sha256')
    if embedded:
        checks['fit_report_sha256']=calibration.get('fit_report_sha256')==embedded
    return dict(status='OK' if all(checks.values()) else 'MISMATCH', checks=checks,
                fit_report_verification='embedded_digest' if embedded else 'not_available_in_capture')


def calibrate_capture(capture):
    raw=Path(capture['path']).read_bytes()
    return calibrate(raw, {}, capture['metadata']['provenance'].get('fit_report_sha256'))


def run(profile_file, capture_paths=None, *, seed=42, count=None, refit=False, limits=None, progress=None):
    profile_file=Path(profile_file)
    raw=profile_file.read_bytes(); profile=json.loads(raw)
    resolve(dict(seed=seed,count=1,output_tokens=420),document=profile)
    if 'joint_distribution' in profile['parameters']:
        resolve(dict(seed=seed,count=1,output_tokens=420,sampling='joint'),document=profile)
    captures=[load_capture(p) for p in (capture_paths if capture_paths is not None else trace_models().values())]
    # Content addressing, never infer fit identity from filenames.
    fit=next((c for c in captures if c['sha256']==profile.get('calibration',{}).get('model_sha256')),None)
    if fit is None and capture_paths != []:
        known={c['sha256'] for c in captures}
        for path in trace_models().values():
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest==profile.get('calibration',{}).get('model_sha256') and digest not in known:
                fit=load_capture(path);captures.insert(0,fit);break
    identity=audit(profile,fit)
    rows=[]
    for cap in captures:
        selected=calibrate_capture(cap) if refit else profile
        selected_identity=audit(selected,cap) if refit else identity
        methods={}
        for method in ('independent','joint'):
            if method=='joint' and 'joint_distribution' not in selected['parameters']:
                methods[method]=dict(status='UNAVAILABLE', reason='profile has no joint distribution; recalibrate explicitly')
                continue
            parameters=dict(seed=seed,count=count or len(cap['events']),output_tokens=420,sampling=method)
            if progress:
                progress(f"{cap['name']}: {method}, {parameters['count']} requests")
            resolved,_=resolve(parameters,document=selected)
            shape=generated_shape(iter_requests(resolved,'fidelity'))
            metrics=compare(cap['shape'],shape)
            methods[method]=dict(status='MEASURED',summary=summarize(shape),metrics=metrics,
                                 grades=grade(metrics,limits,selected_identity['status']=='OK'),
                                 parameters=parameters)
        rows.append(dict(capture=cap['name'],capture_path=cap['path'],capture_sha256=cap['sha256'],
                         fit_sha256=selected['calibration']['model_sha256'],
                         fit_parameters=selected['parameters'],
                         fit_profile_sha256=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
                         role='fit_source' if refit or cap is fit else 'other_window',
                         interpretation='in_sample' if refit or cap is fit else 'cross_window_drift_expected_not_profile_bug',
                         audit=selected_identity,real=summarize(cap['shape']),methods=methods))
    return dict(schema_version=1,profile=str(profile_file.resolve()),profile_sha256=hashlib.sha256(raw).hexdigest(),
                profile_schema=profile.get('schema_version',1),seed=seed,requested_count=count,
                mode='refit_each_capture' if refit else 'fixed_profile',thresholds=thresholds(limits),bins=BINS,
                identity=identity,profile_validation='VALID_PARAMETERS_ONLY',fit_source=fit['name'] if fit else None,
                held_out_validated=profile.get('calibration',{}).get('held_out_validated'),
                limitations=profile.get('calibration',{}).get('limitations',[]),
                verdict_scope='Input-distribution diagnostics only; cache experiments need trace/control validation. Output lengths and temporal reuse are not validated.',
                profile_parameters=profile['parameters'],rows=rows)
