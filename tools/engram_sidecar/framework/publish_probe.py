#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish every row of the deterministic Engram fixture through the real path.

Generates the fixture checkpoint, allocates fabric memory through
engram_publish.publish_shard, and checks that the recorded row ranges and
sha256 digests match a plain read of the same safetensors rows. Needs a GPU and
the native engine; run it from the serving checkout.

    python3 tools/engram_sidecar/framework/publish_probe.py
"""
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from checkpoint_fixture import generate_fixture
from rtp_llm.models_py.modules.dsv41.engram_publish import publish_shard, release
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards

base = Path('/tmp/engram-publish-probe')
if not base.exists():
    generate_fixture(base)
root = base / 'owner_checkpoint'
generation = str(uuid4())
docs = []
for shard in range(1):
    document = publish_shard(root, 'fixture-rev', '11.13.195.90#%d' % shard,
                             generation, shard, 1, device=0, numa=0)
    docs.append(document)

for shard, document in enumerate(docs):
    assert document['shard'] == shard and document['shards'] == 1
    assert document['generation'] == generation
    assert document['host_id'] == '11.13.195.90#%d' % shard
    assert len(document['tensors']) == 4, document['tensors'].keys()
    for name, tensor in document['tensors'].items():
        assert tensor['handle'] and tensor['allocation_id']
        assert tensor['row_begin'] < tensor['row_end']
        assert tensor['logical_bytes'] == (tensor['row_end'] - tensor['row_begin']) * tensor['shape'][1]
        assert tensor['sha256'].startswith('sha256:')

# A single holder covers every row of each tensor.
for name, tensor in docs[0]['tensors'].items():
    assert tensor['row_begin'] == 0
    assert tensor['row_end'] == tensor['shape'][0]

# The recorded digest must match a plain read of the same byte range.
from safetensors import safe_open
index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
for name, tensor in docs[0]['tensors'].items():
    path = root / index[name]
    with safe_open(str(path), framework='pt') as handle:
        rows = handle.get_slice(name)[tensor['row_begin']:tensor['row_end']]
        raw = rows.contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()
    assert 'sha256:' + hashlib.sha256(raw).hexdigest() == tensor['sha256'], name

print('PROBE_OK tensors=%d bytes=%d' % (
    sum(len(d['tensors']) for d in docs),
    sum(t['logical_bytes'] for d in docs for t in d['tensors'].values())))
release(generation)
print('RELEASED')