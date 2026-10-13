"""Discover, audit and compare captures using the production request generator."""
import hashlib
import json
from pathlib import Path

from analysis.traffic_fidelity import compare, summarize, grade, thresholds, BINS
from traffic.catalog import trace_models
from traffic.codecs import decode_model
from traffic.structure import model_shape, generated_shape
from traffic.synthetic.derivation import derive_parameters
from traffic.synthetic.configuration import resolve
from traffic.synthetic.generator import iter_requests


def load_capture(path):
    path=Path(path); raw=path.read_bytes()
    model = decode_model(raw)
    metadata, events = model.to_wire()
    shape = model_shape(model)
    return dict(path=str(path.resolve()),name=path.stem,sha256=hashlib.sha256(raw).hexdigest(),
                metadata=metadata,events=events,shape=shape)


def audit(profile, capture, *, expected=None):
    if capture is None:
        return dict(status='UNAVAILABLE',checks={},note='No capture matches calibration.model_sha256')
    calibration=profile.get('calibration',{})
    if expected is None:
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
    return derive_parameters(raw, {}, capture['metadata']['provenance'].get('fit_report_sha256'))


def run(profile_file, capture_paths=None, *, seed=42, count=None, refit=False, limits=None, progress=None):
    """An explicit empty capture_paths disables discovery and fit-source fallback."""
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
        selected_identity=audit(selected,cap,expected=selected) if refit else identity
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
                         fit_sha256=selected.get('calibration', {}).get('model_sha256'),
                         fit_parameters=selected['parameters'],
                         fit_profile_sha256=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
                         role='fit_source' if refit or cap is fit else 'other_window',
                         interpretation='in_sample' if refit or cap is fit else 'cross_window_drift_expected_not_profile_bug',
                         audit=selected_identity,real=summarize(cap['shape']),methods=methods))
    return dict(traffic_fidelity_schema_version=1,profile=str(profile_file.resolve()),profile_sha256=hashlib.sha256(raw).hexdigest(),
                profile_schema=profile.get('schema_version',1),seed=seed,requested_count=count,
                mode='refit_each_capture' if refit else 'fixed_profile',thresholds=thresholds(limits),bins=BINS,
                identity=identity,profile_validation='VALID_PARAMETERS_ONLY',fit_source=fit['name'] if fit else None,
                held_out_validated=profile.get('calibration',{}).get('held_out_validated'),
                limitations=profile.get('calibration',{}).get('limitations',[]),
                verdict_scope='Input-distribution diagnostics only; cache experiments need trace/control validation. Output lengths and temporal reuse are not validated.',
                profile_parameters=profile['parameters'],rows=rows)
