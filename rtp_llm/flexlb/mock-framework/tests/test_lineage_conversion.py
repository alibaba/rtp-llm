"""Conversion is offline; production Client keeps a single input codec."""
import importlib.util
import json
from pathlib import Path

import pytest
from mock_framework.input import read_inputs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('convert_lineage', ROOT / 'scripts/convert_lineage.py')
conversion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion)
from traffic.datasets import build_manifest
from traffic.prefix_lineage_v3 import encode
from traffic.output_sampling import output_sampler


def source(tmp_path):
    path = tmp_path / 'source.xz'
    path.write_bytes(encode([[100, 70, -1, 0], [125, 130, 0, 1]], block_size=64))
    path.with_suffix('.manifest.json').write_text(json.dumps(build_manifest(path)))
    return path


def test_exact_lengths_prefixes_arrival_offsets_and_original_output_sampling_survive_conversion(tmp_path):
    original = source(tmp_path)
    target = tmp_path / 'converted.binpb.zst'
    manifest = conversion.convert(original, target, output_cap=8192, output_mean=420, seed=20260913, priority=50)
    rows = list(read_inputs(target))
    assert [len(row.token_ids) for row in rows] == [70, 130]
    assert rows[0].token_ids[:64] == rows[1].token_ids[:64]
    assert rows[0].token_ids[64] != rows[1].token_ids[64]
    assert [row.arrival_offset_ms for row in rows] == [0, 25]
    sampler = output_sampler(dict(output_tokens=8192, output_distribution=dict(kind='geometric', mean_tokens=420, seed=20260913)))
    assert [row.max_new_tokens for row in rows] == [sampler(0), sampler(1)]
    assert manifest['requests'] == 2 and manifest['input_tokens'] == 200
    assert manifest['source_span_ms'] == 25
    before = target.read_bytes()
    with pytest.raises(ValueError, match='fresh'):
        conversion.convert(original, target, output_cap=8192, output_mean=420, seed=1, priority=50)
    assert target.read_bytes() == before


def test_invalid_conversion_removes_partial_output(tmp_path):
    target = tmp_path / 'bad.binpb.zst'
    with pytest.raises(ValueError):
        conversion.convert(source(tmp_path), target, output_cap=0, output_mean=420, seed=1, priority=50)
    assert not target.exists() and not target.with_name('bad.manifest.json').exists()
