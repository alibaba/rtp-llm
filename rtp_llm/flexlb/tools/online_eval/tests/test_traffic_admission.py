import hashlib
import json
import subprocess
import sys
from unittest.mock import patch
import pytest
from traffic import datasets
from traffic.codecs import codec_for_version, decode_model
from traffic.synthetic.derivation import derive_parameters
from traffic.synthetic.configuration import resolve
from traffic.synthetic.generator import iter_requests
from traffic.traffic_source import materialize, validate_plan
from traffic.statistics import statistics
from scripts.pipeline.derive_master_templates import derive
from scripts.pipeline.derive_synthetic_profile import main as derive_profile


@pytest.mark.parametrize('block', [64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize('version', [2, 3])
def test_every_capture_geometry_can_calibrate_generate_and_materialize(tmp_path, block, version):
    raw = codec_for_version(version).encode([[0, block * 2 + 1, -1, 0],
        [1000, block * 3 + 1, 0, 1]], block_size=block)
    model = decode_model(raw)
    profile = derive_parameters(model, {}, None, model_sha256=hashlib.sha256(raw).hexdigest())
    assert profile['calibration']['family_statistic'] == 'lineage_tree_request_count'
    assert 'lineage_tree_top5_share' in profile['calibration']['targets']
    assert statistics(model)['input_tokens']['total'] == sum(e.input_tokens for e in model.events)
    for sampling in ('independent', 'joint'):
        workload, _ = resolve(dict(seed=4, count=12, output_tokens=2, sampling=sampling), document=profile)
        rows = list(iter_requests(workload, 'geometry'))
        path = tmp_path / sampling
        path.write_text('\n'.join(map(json.dumps, rows)) + '\n')
        assert validate_plan(path, 'geometry') == 12
        assert all(row['cache_key_block_size'] == block for row in rows)


def pin(tmp_path):
    model = tmp_path / 'model.xz'
    model.write_bytes(codec_for_version(3).encode([[0, 1025, -1, 0], [1000, 1025, 0, 1]]))
    model.with_suffix('.manifest.json').write_text(json.dumps(datasets.build_manifest(model)))
    return model


def test_admitted_snapshot_is_decoded_once_and_reused_without_reading_source(tmp_path):
    path = pin(tmp_path)
    with patch('traffic.datasets.decode_model', wraps=decode_model) as decode:
        captured = datasets.CapturedDataset.read(path)
        assert decode.call_count == 1
    path.unlink()  # The admitted snapshot is the exact input consumed by this run.
    spec = dict(kind='trace', model='prefix_lineage', parameters=dict(path=path.name,
        count=2, output_tokens=3, priority=50))
    with patch('traffic.traffic_source.decode_model', side_effect=AssertionError('second decode')):
        plan = materialize(tmp_path / 'plan', spec, 'snapshot', tmp_path, captured=captured)
    assert validate_plan(plan) == 2
    manifest = json.loads(plan.with_suffix('.manifest.json').read_text())
    assert manifest['model_sha256'] == captured.sha256
    for bad in (dict(spec, version='2'), dict(spec, parameters=dict(spec['parameters'], path='other.xz'))):
        with pytest.raises(ValueError):
            materialize(tmp_path / 'bad', bad, 'snapshot', tmp_path, captured=captured)
    assert not (tmp_path / 'bad').exists()


def test_template_and_profile_cli_decode_once(tmp_path):
    path = pin(tmp_path)
    with patch('traffic.datasets.decode_model', wraps=decode_model) as decode:
        assert len(derive(path, count=2)['templates']) == 2
        assert decode.call_count == 1
    report = tmp_path / 'fit.json'
    report.write_text('{}')
    output = tmp_path / 'profile.json'
    with patch('traffic.datasets.decode_model', wraps=decode_model) as decode:
        derive_profile(['--model', str(path), '--fit-report', str(report), '--out', str(output)])
        assert decode.call_count == 1
    profile = json.loads(output.read_text())
    assert profile['calibration']['model_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_content_configuration_does_not_load_playback_or_codecs():
    subprocess.run([sys.executable, '-c',
        "import sys; import traffic.synthetic.configuration; "
        "assert 'traffic.playback.configuration' not in sys.modules; "
        "assert 'traffic.output_sampling' not in sys.modules; "
        "assert 'traffic.codecs' not in sys.modules"], check=True)
