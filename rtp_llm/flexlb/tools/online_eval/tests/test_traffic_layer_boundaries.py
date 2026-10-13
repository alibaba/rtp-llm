"""Cross-version geometry, domain expansion and sender evidence boundaries."""
import json
import tempfile
from pathlib import Path
import pytest
from traffic.codecs import decode_model, codec_for_version
from traffic.lineage_model import LineageExpander
from traffic.structure import model_shape
from traffic.catalog import trace_models
from analysis.traffic_fidelity import load_capture
from adapters.playback_evidence import PlaybackEvidence


def test_all_pinned_capture_versions_load_for_fidelity():
    captures = [load_capture(path) for path in trace_models().values()]
    assert {c['metadata']['version'] for c in captures} == {2, 3}
    assert all(len(c['shape']['blocks']) == c['metadata']['count'] for c in captures)


@pytest.mark.parametrize('version', [2, 3])
@pytest.mark.parametrize('block', [64, 1024, 4096])
def test_retained_expansion_preserves_full_label_allocation(version, block):
    raw = codec_for_version(version).encode([[0, block * 90 + 1, -1, 0],
        [1, block * 91 + 1, 0, 89], [2, block * 2, -1, 0]], block_size=block)
    model = decode_model(raw)
    width = 4096 // block
    full = list(LineageExpander(model))
    retained = list(LineageExpander(model, retained_blocks=width))
    assert [r.block_labels[:width] for r in full] == [r.block_labels for r in retained]
    assert model_shape(model)['blocks'] == [r.input_tokens / 512 for r in full]


def test_java_playback_evidence_checks_launch_and_preserves_reproduction_fields():
    row = dict(trace_sha256='frozen', offline_plan_supported=True, planner_version='PLAYBACK_LEGACY_V1',
        token_stride=18, mode='replay', identity='structural-relabel', seed=0, max_laps=1,
        num_shards=1, shard_index=0, duration_seconds=0, limit=0, qps=0, speed=1,
        ramp_up_seconds=0, retain_probability=0, burst_factor=1, burst_period_seconds=10,
        burst_duty=.1, diurnal_amplitude=0, diurnal_period_seconds=86400)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'playback.json'
        path.write_text(json.dumps(row))
        assert PlaybackEvidence.read(path, 'frozen', {}).to_dict() == row
        for field, value in [('trace_sha256', 'different'), ('seed', 5), ('identity', 'none'),
                             ('offline_plan_supported', False), ('token_stride', 0)]:
            path.write_text(json.dumps(dict(row, **{field: value})))
            with pytest.raises(ValueError):
                PlaybackEvidence.read(path, 'frozen', {})
