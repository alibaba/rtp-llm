"""New schemas preserve old immutable data and reject ambiguous evidence."""
import copy
import hashlib
import json
import pytest
from traffic.datasets import build_manifest, read_manifest, statistics
from traffic.manifest_contract import layered, normalize
from traffic.prefix_lineage import encode
from artifacts.performance_capture import whale_read_plan, write_manifest, read_manifest as read_capture, LAYERS, ROLES


def test_layered_roundtrip_preserves_facts_and_rejects_drift(tmp_path):
    model = tmp_path / 'test.xz'
    model.write_bytes(encode([[0,512,-1,0],[1000,1024,0,1]]))
    old = build_manifest(model)
    old['limitations'].append('independently recorded capture problem')
    new = layered(old, previous=old)
    assert normalize(new) == old
    model.with_suffix('.manifest.json').write_text(json.dumps(new))
    assert read_manifest(model) == old
    new['artifact']['codec']['version'] = 99
    model.with_suffix('.manifest.json').write_text(json.dumps(new))
    with pytest.raises(ValueError, match='codec'):
        read_manifest(model)
    new = layered(old)
    new['capture']['requested_window']['start_epoch_ms'] = 123
    with pytest.raises(ValueError, match='window'):
        normalize(new)


def test_statistics_uses_the_data_block_size():
    s = statistics([[0,2,-1,0],[1000,3,0,1]], version=2, block_size=1024)
    assert s['input_tokens']['total'] == 5120
    assert s['prefix_structure']['shared_tokens']['max'] == 1024


def test_read_plan_pins_deployed_versions():
    deployment = dict(id='deployment',py_biz_id='biz',biz_version=17,zone_deployments=[dict(zone_name='prefill')])
    images = dict(deployment_id='deployment',biz_id='biz',biz_version=17,template_version_id='pinned-template')
    plan = whale_read_plan(deployment,images)
    assert plan[0]['parameters']['id'] == 'pinned-template'
    assert plan[1]['parameters']['biz_version'] == '17'
    assert all('latest' not in row['operation'].lower() for row in plan)
    with pytest.raises(ValueError,match='disagree'):
        whale_read_plan(deployment,dict(images,biz_version=18))


def capture_document(tmp_path):
    evidence = tmp_path/'evidence.json';evidence.write_text('{}')
    ref = dict(path=evidence.name,sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    unavailable = dict(status='unavailable',reason='not captured',evidence=None,
        roles={role:dict(status='unavailable',image=None,commit=None,version_basis=None) for role in ROLES})
    return dict(performance_capture_schema_version=1,
        identity=dict(id='fixture',model='model',hardware='hardware',deployment_ref='opaque:fixture'),
        collection=dict(status='partial',started_at='2026-01-01T00:00:00+00:00',finished_at='2026-01-01T00:01:00+00:00',tool='fixture',issues=['missing process readback']),
        configuration={layer:copy.deepcopy(unavailable) for layer in LAYERS},measurements={},derived={}),ref


def test_capture_references_and_availability(tmp_path):
    document,ref=capture_document(tmp_path)
    document['configuration']['deployment'].update(status='captured',reason=None,evidence=ref)
    document['measurements']['prefill']=dict(evidence=ref,configuration_layers=['deployment'],method='fixture timing',unit='ms')
    document['derived']['mock']=dict(evidence=ref,measurement_ids=['prefill'],method='fixture fit')
    manifest=tmp_path/'manifest.json';write_manifest(manifest,document)
    assert read_capture(manifest)==document
    with pytest.raises(ValueError,match='overwrite'):
        write_manifest(manifest,document)
    (tmp_path/'evidence.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checksum'):
        read_capture(manifest)


def test_declared_image_cannot_claim_process_verification(tmp_path):
    document,ref=capture_document(tmp_path)
    layer=document['configuration']['template'];layer.update(status='captured',reason=None,evidence=ref)
    layer['roles']['prefill']=dict(status='recorded',image='candidate',commit='short-tag',version_basis='process_readback')
    with pytest.raises(ValueError,match='runtime verification'):
        write_manifest(tmp_path/'manifest.json',document)


def test_capture_reference_cannot_escape_bundle(tmp_path):
    document,ref=capture_document(tmp_path)
    document['configuration']['template'].update(status='captured',reason=None,evidence=dict(ref,path='../secret.json'))
    with pytest.raises(ValueError,match='escapes bundle'):
        write_manifest(tmp_path/'manifest.json',document)
