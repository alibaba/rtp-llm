"""Cross-version geometry, domain expansion and sender evidence boundaries."""
import json
from traffic.playback.defaults import DEFAULTS
import tempfile
from pathlib import Path
import pytest
from traffic.codecs import decode_model, codec_for_version
from traffic.lineage_model import LineageExpander
from traffic.structure import model_shape
from traffic.catalog import trace_models
from diagnostics.traffic_fidelity import load_capture
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
    row = dict(trace_sha256='frozen', offline_plan_supported=True, planner_version='PLAYBACK_V1',
        token_stride=18, mode='replay', identity='structural-relabel', seed=0, max_laps=1,
        num_shards=1, shard_index=0, duration_seconds=0, limit=0, qps=0, speed=float(DEFAULTS["REPLAY_SPEED"]),
        ramp_up_seconds=0, retain_probability=0, burst_factor=1, burst_period_seconds=10,
        burst_duty=.1, diurnal_amplitude=0, diurnal_period_seconds=86400)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'playback.json'
        path.write_text(json.dumps(row))
        assert PlaybackEvidence.read(path, 'frozen', {}).to_dict() == row
        for field, value in [('trace_sha256', 'different'), ('max_laps', 5), ('identity', 'none'),
                             ('offline_plan_supported', False), ('token_stride', 0)]:
            path.write_text(json.dumps(dict(row, **{field: value})))
            with pytest.raises(ValueError):
                PlaybackEvidence.read(path, 'frozen', {})


def test_fidelity_refits_the_loaded_snapshot_after_capture_changes(tmp_path):
    from diagnostics.traffic_fidelity import calibrate_capture
    from traffic.prefix_lineage import encode
    path = tmp_path / 'capture.xz'
    path.write_bytes(encode([[0, 512, -1, 0], [1, 1024, -1, 0]], {}))
    capture = load_capture(path)
    expected = calibrate_capture(capture)
    path.write_bytes(encode([[0, 1024, -1, 0], [1, 2048, -1, 0]], {}))
    assert calibrate_capture(capture) == expected
    path.unlink()
    assert calibrate_capture(capture) == expected


def test_token_plan_normalizes_compact_and_explicit_blocks_identically(tmp_path):
    from traffic.token_plan import read_token_plan
    from analysis.workload_profile import profile
    path = tmp_path / 'tokens.jsonl'
    rows = [dict(ts=0, il=5, cache_key_block_size=2, input_ids=[1, 1, 2, 2, 3]),
            dict(ts=1, il=5, cache_key_block_size=2, input_token_blocks=[1, [2, 2], 3])]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    first, second = read_token_plan(path)
    assert first.prefix_keys == second.prefix_keys
    assert first.family == second.family == 'unknown'
    # The partial final block contributes input tokens, but no cache-key hit.
    assert profile(iter([first, second]), (2,))['lru_hit_by_capacity']['2'] == .4


def test_analysis_profiles_a_one_pass_iterator_without_a_file():
    from traffic.token_plan import TokenPlanRequest
    from analysis.workload_profile import profile
    requests = iter([TokenPlanRequest(0, 4, 2, (b'a', b'b'), 'shared'),
                     TokenPlanRequest(1, 4, 2, (b'a', b'b'), 'shared')])
    assert profile(requests, (2,))['infinite_cache_potential_hit'] == .5
    assert list(requests) == []
